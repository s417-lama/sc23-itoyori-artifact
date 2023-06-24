#pragma once

#include <cstddef>
#include <cstdlib>
#include <list>
#include <unordered_map>
#include <optional>
#include <sys/mman.h>
#include <mpi.h>

#define PCAS_HAS_MEMORY_RESOURCE __has_include(<memory_resource>)
#if PCAS_HAS_MEMORY_RESOURCE
#include <memory_resource>
namespace pcas { namespace pmr = std::pmr; }
#else
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/unsynchronized_pool_resource.hpp>
#include <boost/container/pmr/pool_options.hpp>
namespace pcas { namespace pmr = boost::container::pmr; }
#endif

#include "pcas/util.hpp"
#include "pcas/logger/logger.hpp"
#include "pcas/topology.hpp"
#include "pcas/virtual_mem.hpp"
#include "pcas/physical_mem.hpp"

namespace pcas {

template <typename P>
class allocator_if final : public pmr::memory_resource {
protected:
  const topology& topo_;

  const std::size_t local_max_size_;
  const std::size_t global_max_size_;

  virtual_mem vm_;
  physical_mem pm_;

  win_manager win_;

  typename P::template allocator_impl_t<P> allocator_;

  std::size_t get_local_max_size() const {
    std::size_t upper_limit = (std::size_t(1) << 44) / next_pow2(topo_.global_nproc());
    std::size_t default_local_size;
    if constexpr (P::use_mpi_win_dynamic) {
      default_local_size = upper_limit;
    } else {
      default_local_size = std::size_t(128) * 1024 * 1024;
    }

    auto ret = std::size_t(getenv_coll("PCAS_ALLOCATOR_MAX_LOCAL_SIZE", default_local_size / 1024 / 1024, topo_.global_comm())) * 1024 * 1024; // MB
    PCAS_CHECK(ret <= upper_limit);
    return ret;
  }

  static std::string allocator_shmem_name(int inter_rank) {
    std::stringstream ss;
    ss << "/pcas_allocator_" << inter_rank;
    return ss.str();
  }

  physical_mem init_pm() const {
    physical_mem pm;

    if (topo_.intra_rank() == 0) {
      pm = physical_mem(allocator_shmem_name(topo_.inter_rank()), global_max_size_, true, false);
    }

    MPI_Barrier(topo_.intra_comm());

    if (topo_.intra_rank() != 0) {
      pm = physical_mem(allocator_shmem_name(topo_.inter_rank()), global_max_size_, false, false);
    }

    PCAS_CHECK(vm_.size() == global_max_size_);
    pm.map(vm_.addr(), 0, vm_.size());

    return pm;
  }

  win_manager create_win() const {
    if constexpr (P::use_mpi_win_dynamic) {
      return win_manager{topo_.global_comm()};
    } else {
      auto local_base_addr = reinterpret_cast<std::byte*>(vm_.addr()) + local_max_size_ * topo_.global_rank();
      return win_manager{topo_.global_comm(), local_base_addr, local_max_size_};
    }
  }

public:
  allocator_if(const topology& topo) :
    topo_(topo),
    local_max_size_(get_local_max_size()),
    global_max_size_(local_max_size_ * topo_.global_nproc()),
    vm_(reserve_same_vm_coll(topo.global_comm(), global_max_size_, local_max_size_)),
    pm_(init_pm()),
    win_(create_win()),
    allocator_(topo_, vm_, win_.win()) {}

  MPI_Win win() const { return win_.win(); }

  bool belongs_to(const void* p) {
    return vm_.addr() <= p && p < reinterpret_cast<std::byte*>(vm_.addr()) + global_max_size_;
  }

  topology::rank_t get_owner(const void* p) const {
    return (reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(vm_.addr())) / local_max_size_;
  }

  std::size_t get_disp(const void* p) const {
    if constexpr (P::use_mpi_win_dynamic) {
      return reinterpret_cast<uintptr_t>(p);
    } else {
      return (reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(vm_.addr())) % local_max_size_;
    }
  }

  void* do_allocate(std::size_t bytes, std::size_t alignment = alignof(max_align_t)) override {
    return allocator_.do_allocate(bytes, alignment);
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(max_align_t)) override {
    allocator_.do_deallocate(p, bytes, alignment);
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  void remote_deallocate(void* p, std::size_t bytes, int target_rank, std::size_t alignment = alignof(max_align_t)) {
    allocator_.remote_deallocate(p, bytes, target_rank, alignment);
  }

  void collect_deallocated() {
    allocator_.collect_deallocated();
  }

  // mainly for debugging
  bool empty() {
    return allocator_.empty();
  }
};

template <typename P>
class mpi_win_resource final : public pmr::memory_resource {
  const topology&   topo_;
  const std::size_t local_max_size_;
  const std::size_t local_base_addr_;
  const MPI_Win     win_;

  std::list<span> freelist_;

public:
  mpi_win_resource(const topology& topo,
                   std::size_t     local_max_size,
                   std::size_t     local_base_addr,
                   MPI_Win         win) :
    topo_(topo),
    local_max_size_(local_max_size),
    local_base_addr_(local_base_addr),
    win_(win),
    freelist_(1, {local_base_addr, local_max_size}) {}

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (alignment > P::block_size) {
      die("Alignment request for allocation must be <= %ld (block size)", P::block_size);
    }

    // Align with block size
    std::size_t real_bytes = round_up_pow2(bytes, P::block_size);

    // FIXME: assumption that freelist returns block-aligned address
    auto s = freelist_get(freelist_, real_bytes);
    if (!s.has_value()) {
      die("Could not allocate memory for malloc_local()");
    }
    PCAS_CHECK(s->size == real_bytes);
    PCAS_CHECK(s->addr % P::block_size == 0);

    void* ret = reinterpret_cast<void*>(s->addr);

    if constexpr (P::use_mpi_win_dynamic) {
      MPI_Win_attach(win_, ret, s->size);
    }

    return ret;
  }

  void do_deallocate(void* p, std::size_t bytes, [[maybe_unused]] std::size_t alignment) override {
    std::size_t real_bytes = round_up_pow2(bytes, P::block_size);
    span s {reinterpret_cast<std::size_t>(p), real_bytes};

    PCAS_CHECK(reinterpret_cast<std::size_t>(p) % P::block_size == 0);

    if constexpr (P::use_mpi_win_dynamic) {
      MPI_Win_detach(win_, p);

      if (madvise(p, real_bytes, MADV_REMOVE) == -1) {
        perror("madvise");
        die("madvise() failed");
      }
    }

    freelist_add(freelist_, s);
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};

class block_resource : public pmr::memory_resource {
  pmr::memory_resource* upstream_mr_;
  const std::size_t     block_size_;

  std::list<span> freelist_;

public:
  block_resource(pmr::memory_resource* upstream_mr,
                 std::size_t           block_size) :
    upstream_mr_(upstream_mr),
    block_size_(block_size) {}

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (bytes >= block_size_) {
      return upstream_mr_->allocate(bytes, alignment);
    }

    std::size_t real_bytes = bytes + alignment;

    auto s = freelist_get(freelist_, real_bytes);
    if (!s.has_value()) {
      void* new_block = upstream_mr_->allocate(block_size_);
      freelist_add(freelist_, {reinterpret_cast<std::size_t>(new_block), block_size_});
      s = freelist_get(freelist_, real_bytes);
      PCAS_CHECK(s.has_value());
    }
    PCAS_CHECK(s->size == real_bytes);

    auto addr = round_up_pow2(s->addr, alignment);

    PCAS_CHECK(addr >= s->addr);
    PCAS_CHECK(s->addr + s->size >= addr + bytes);

    freelist_add(freelist_, {s->addr, addr - s->addr});
    freelist_add(freelist_, {addr + bytes, s->addr + s->size - (addr + bytes)});

    return reinterpret_cast<void*>(addr);
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    if (bytes >= block_size_) {
      upstream_mr_->deallocate(p, bytes, alignment);
    }

    freelist_add(freelist_, {reinterpret_cast<std::size_t>(p), bytes});
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }
};

template <typename P>
class std_pool_resource_impl {
  const topology&                   topo_;
  const virtual_mem&                vm_;
  const std::size_t                 local_max_size_;
  const std::size_t                 local_base_addr_;
  MPI_Win                           win_;
  mpi_win_resource<P>               win_mr_;
  block_resource                    block_mr_;
  pmr::unsynchronized_pool_resource mr_;
  int                               max_unflushed_free_objs_;

  using logger = typename P::logger;
  using logger_kind = typename P::logger::kind::value;

  struct header {
    header*     prev      = nullptr;
    header*     next      = nullptr;
    std::size_t size      = 0;
    std::size_t alignment = 0;
    int         freed     = 0;
  };

  header allocated_list_;
  header* allocated_list_end_ = &allocated_list_;

  void remove_header_from_list(header* h) {
    PCAS_CHECK(h->prev);
    h->prev->next = h->next;

    if (h->next) {
      h->next->prev = h->prev;
    } else {
      PCAS_CHECK(h == allocated_list_end_);
      allocated_list_end_ = h->prev;
    }
  }

  std::size_t get_header_disp(const void* p, std::size_t alignment) const {
    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    auto h = reinterpret_cast<const header*>(reinterpret_cast<const std::byte*>(p) - pad_bytes);
    const void* flag_addr = &h->freed;

    if constexpr (P::use_mpi_win_dynamic) {
      return reinterpret_cast<uintptr_t>(flag_addr);
    } else {
      return (reinterpret_cast<uintptr_t>(flag_addr) - reinterpret_cast<uintptr_t>(vm_.addr())) % local_max_size_;
    }
  }

  // FIXME: workaround for boost
  // Ideally: pmr::pool_options{.max_blocks_per_chunk = (std::size_t)16 * 1024 * 1024 * 1024}
  pmr::pool_options my_pool_options() {
    pmr::pool_options opts;
    opts.max_blocks_per_chunk = std::size_t(16) * 1024 * 1024 * 1024;
    return opts;
  }

public:
  std_pool_resource_impl(const topology&    topo,
                         const virtual_mem& vm,
                         MPI_Win            win) :
    topo_(topo),
    vm_(vm),
    local_max_size_(vm_.size() / topo_.global_nproc()),
    local_base_addr_(reinterpret_cast<uintptr_t>(vm_.addr()) + local_max_size_ * topo_.global_rank()),
    win_(win),
    win_mr_(topo, local_max_size_, local_base_addr_, win),
    block_mr_(&win_mr_, std::size_t(getenv_coll("PCAS_ALLOCATOR_BLOCK_SIZE", 2, topo_.global_comm())) * 1024 * 1024),
    mr_(my_pool_options(), &block_mr_),
    max_unflushed_free_objs_(getenv_coll("PCAS_ALLOCATOR_MAX_UNFLUSHED_FREE_OBJS", 10, topo_.global_comm())) {}

  void* do_allocate(std::size_t bytes, std::size_t alignment) {
    auto ev = logger::template record<logger_kind::MemAlloc>(bytes);

    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    std::size_t real_bytes = bytes + pad_bytes;

    std::byte* p = reinterpret_cast<std::byte*>(mr_.allocate(real_bytes, alignment));
    std::byte* ret = p + pad_bytes;

    PCAS_CHECK(ret + bytes <= p + real_bytes);
    PCAS_CHECK(p + sizeof(header) <= ret);

    header* h = new(p) header {
      .prev = allocated_list_end_, .next = nullptr,
      .size = real_bytes, .alignment = alignment, .freed = 0};
    PCAS_CHECK(allocated_list_end_->next == nullptr);
    allocated_list_end_->next = h;
    allocated_list_end_ = h;

    return ret;
  }

  void do_deallocate(void* p, std::size_t bytes, [[maybe_unused]] std::size_t alignment) {
    auto ev = logger::template record<logger_kind::MemFree>(bytes);

    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    std::size_t real_bytes = bytes + pad_bytes;

    header* h = reinterpret_cast<header*>(reinterpret_cast<std::byte*>(p) - pad_bytes);
    remove_header_from_list(h);

    mr_.deallocate(h, real_bytes, alignment);
  }

  void remote_deallocate(void* p, std::size_t bytes [[maybe_unused]], int target_rank, std::size_t alignment) {
    PCAS_CHECK(topo_.global_rank() != target_rank);

    static constexpr int one = 1;
    static int ret; // dummy value; passing NULL to result_addr causes segfault on some MPI
    MPI_Fetch_and_op(&one,
                     &ret,
                     MPI_INT,
                     target_rank,
                     get_header_disp(p, alignment),
                     MPI_REPLACE,
                     win_);

    static int count = 0;
    count++;
    if (count >= max_unflushed_free_objs_) {
      MPI_Win_flush_all(win_);
      count = 0;
    }
  }

  void collect_deallocated() {
    auto ev = logger::template record<logger_kind::MemCollect>();

    header *h = allocated_list_.next;
    while (h) {
      if (h->freed) {
        header h_copy = *h;
        remove_header_from_list(h);
        mr_.deallocate((void*)h, h_copy.size, h_copy.alignment);
        h = h_copy.next;
      } else {
        h = h->next;
      }
    }
  }

  bool empty() {
    return allocated_list_.next == nullptr;
  }
};

// Policy
// -----------------------------------------------------------------------------

struct allocator_policy_default {
  constexpr static bool use_mpi_win_dynamic = true;
  constexpr static std::size_t block_size = 65536;
  using logger = logger::logger_if<logger::policy_default>;
  template <typename P>
  using allocator_impl_t = std_pool_resource_impl<P>;
};

// Tests
// -----------------------------------------------------------------------------

PCAS_TEST_CASE("[pcas::allocator] basic test") {
  allocator_if<allocator_policy_default> allocator(MPI_COMM_WORLD);

  PCAS_SUBCASE("Local alloc/dealloc") {
    std::vector<std::size_t> sizes = {1, 2, 4, 8, 16, 32, 100, 200, 1000, 100000, 1000000};
    constexpr int N = 10;
    for (auto size : sizes) {
      void* ptrs[N];
      for (int i = 0; i < N; i++) {
        ptrs[i] = allocator.allocate(size);
        for (std::size_t j = 0; j < size; j += 128) {
          reinterpret_cast<char*>(ptrs[i])[j] = 0;
        }
      }
      for (int i = 0; i < N; i++) {
        allocator.deallocate(ptrs[i], size);
      }
    }
  }

  PCAS_SUBCASE("Remote access") {
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    std::size_t size = 128;
    void* p = allocator.allocate(size);

    for (std::size_t i = 0; i < size; i++) {
      reinterpret_cast<uint8_t*>(p)[i] = rank;
    }

    std::vector<uint64_t> addrs(nproc);
    addrs[rank] = reinterpret_cast<uint64_t>(p);

    // GET
    for (int target_rank = 0; target_rank < nproc; target_rank++) {
      MPI_Bcast(&addrs[target_rank], 1, MPI_UINT64_T, target_rank, MPI_COMM_WORLD);
      if (rank != target_rank) {
        std::vector<uint8_t> buf(size);
        MPI_Get(buf.data(),
                size,
                MPI_BYTE,
                target_rank,
                addrs[target_rank],
                size,
                MPI_BYTE,
                allocator.win());
        MPI_Win_flush(target_rank, allocator.win());

        for (std::size_t i = 0; i < size; i++) {
          PCAS_CHECK(buf[i] == target_rank);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // PUT
    std::vector<uint8_t> buf(size);
    for (std::size_t i = 0; i < size; i++) {
      buf[i] = rank;
    }

    int target_rank = (rank + 1) % nproc;
    MPI_Put(buf.data(),
            size,
            MPI_UINT8_T,
            target_rank,
            addrs[target_rank],
            size,
            MPI_UINT8_T,
            allocator.win());
    MPI_Win_flush(target_rank, allocator.win());

    MPI_Barrier(MPI_COMM_WORLD);

    for (std::size_t i = 0; i < size; i++) {
      PCAS_CHECK(reinterpret_cast<uint8_t*>(p)[i] == (nproc + rank - 1) % nproc);
    }

    PCAS_SUBCASE("Local free") {
      allocator.deallocate(p, size);
    }

    if (nproc > 1) {
      PCAS_SUBCASE("Remote free") {
        PCAS_CHECK(!allocator.empty());

        MPI_Barrier(MPI_COMM_WORLD);

        int target_rank = (rank + 1) % nproc;
        allocator.remote_deallocate(reinterpret_cast<void*>(addrs[target_rank]), size, target_rank);

        MPI_Win_flush_all(allocator.win());
        MPI_Barrier(MPI_COMM_WORLD);

        allocator.collect_deallocated();
      }
    }

    PCAS_CHECK(allocator.empty());
  }
}

}
