#pragma once

#include <mpi.h>

#include "pcas/util.hpp"

namespace pcas {

class topology {
public:
  using rank_t = int;

private:
  struct comm_group {
    rank_t   rank  = -1;
    rank_t   nproc = -1;
    MPI_Comm comm  = MPI_COMM_NULL;
    bool     own   = false;

    comm_group(MPI_Comm c, bool own) : comm(c), own(own) {
      MPI_Comm_rank(c, &rank);
      MPI_Comm_size(c, &nproc);
      PCAS_CHECK(rank != -1);
      PCAS_CHECK(nproc != -1);
    }

    ~comm_group() {
      if (own) {
        MPI_Comm_free(&comm);
      }
    }
  };

  const comm_group cg_global_;

  const bool shared_memory_enabled_;

  const comm_group cg_intra_;
  const comm_group cg_inter_;

  struct map_entry {
    rank_t intra_rank;
    rank_t inter_rank;
  };
  const std::vector<map_entry> process_map_; // global_rank -> (intra, inter rank)
  const std::vector<rank_t> intra2global_rank_;

  MPI_Comm create_intra_comm() {
    if (shared_memory_enabled_) {
      MPI_Comm h;
      MPI_Comm_split_type(global_comm(), MPI_COMM_TYPE_SHARED, global_rank(), MPI_INFO_NULL, &h);
      return h;
    } else {
      return MPI_COMM_SELF;
    }
  }

  MPI_Comm create_inter_comm() {
    if (shared_memory_enabled_) {
      MPI_Comm h;
      MPI_Comm_split(global_comm(), intra_rank(), global_rank(), &h);
      return h;
    } else {
      return global_comm();
    }
  }

  std::vector<map_entry> create_process_map() {
    map_entry my_entry {intra_rank(), inter_rank()};
    std::vector<map_entry> ret(global_nproc());
    MPI_Allgather(&my_entry,
                  sizeof(map_entry),
                  MPI_BYTE,
                  ret.data(),
                  sizeof(map_entry),
                  MPI_BYTE,
                  global_comm());
    return ret;
  }

  std::vector<rank_t> create_intra2global_rank() {
    std::vector<rank_t> ret;
    for (rank_t i = 0; i < global_nproc(); i++) {
      if (process_map_[i].inter_rank == inter_rank()) {
        ret.push_back(i);
      }
    }
    PCAS_CHECK(ret.size() == static_cast<std::size_t>(intra_nproc()));
    return ret;
  }

public:
  topology(MPI_Comm comm) :
    cg_global_(comm, false),
    shared_memory_enabled_(getenv_coll("PCAS_ENABLE_SHARED_MEMORY", 1, global_comm())),
    cg_intra_(create_intra_comm(), shared_memory_enabled_),
    cg_inter_(create_inter_comm(), shared_memory_enabled_),
    process_map_(create_process_map()),
    intra2global_rank_(create_intra2global_rank()) {}

  MPI_Comm global_comm()  const { return cg_global_.comm;  }
  rank_t   global_rank()  const { return cg_global_.rank;  }
  rank_t   global_nproc() const { return cg_global_.nproc; }

  MPI_Comm intra_comm()   const { return cg_intra_.comm;   }
  rank_t   intra_rank()   const { return cg_intra_.rank;   }
  rank_t   intra_nproc()  const { return cg_intra_.nproc;  }

  MPI_Comm inter_comm()   const { return cg_inter_.comm;   }
  rank_t   inter_rank()   const { return cg_inter_.rank;   }
  rank_t   inter_nproc()  const { return cg_inter_.nproc;  }

  rank_t intra_rank(rank_t global_rank) const {
    PCAS_CHECK(0 <= global_rank);
    PCAS_CHECK(global_rank < global_nproc());
    return process_map_[global_rank].intra_rank;
  }

  rank_t inter_rank(rank_t global_rank) const {
    PCAS_CHECK(0 <= global_rank);
    PCAS_CHECK(global_rank < global_nproc());
    return process_map_[global_rank].inter_rank;
  }

  rank_t intra2global_rank(rank_t intra_rank) const {
    PCAS_CHECK(0 <= intra_rank);
    PCAS_CHECK(intra_rank < intra_nproc());
    return intra2global_rank_[intra_rank];
  }

  bool is_locally_accessible(topology::rank_t target_global_rank) const {
    return inter_rank(target_global_rank) == inter_rank();
  }

};

}
