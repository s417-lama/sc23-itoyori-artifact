/*
 *         ---- The Unbalanced Tree Search (UTS) Benchmark ----
 *  
 *  Copyright (c) 2010 See AUTHORS file for copyright holders
 *
 *  This file is part of the unbalanced tree search benchmark.  This
 *  project is licensed under the MIT Open Source license.  See the LICENSE
 *  file for copyright and licensing information.
 *
 *  UTS is a collaborative project between researchers at the University of
 *  Maryland, the University of North Carolina at Chapel Hill, and the Ohio
 *  State University.  See AUTHORS file for more information.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <new>

#include "ityr/ityr.hpp"

using my_ityr = ityr::ityr_if<ityr::ityr_policy>;
template <typename T>
using global_ptr = my_ityr::global_ptr<T>;

#include "uts.h"

#ifndef UTS_USE_VECTOR
#define UTS_USE_VECTOR 0
#endif

#ifndef UTS_REBUILD_TREE
#define UTS_REBUILD_TREE 0
#endif

#ifndef UTS_RUN_SEQ
#define UTS_RUN_SEQ 0
#endif

#ifndef UTS_RECURSIVE_FOR
#define UTS_RECURSIVE_FOR 0
#endif

/***********************************************************
 *  UTS Implementation Hooks                               *
 ***********************************************************/

// The name of this implementation
const char * impl_getName(void) {
  return "Itoyori Parallel Search";
}

int impl_paramsToStr(char *strBuf, int ind) {
  ind += sprintf(strBuf + ind, "Execution strategy:  %s\n", impl_getName());
  return ind;
}

// Not using UTS command line params, return non-success
int impl_parseParam(char *param, char *value) {
  return 1;
}

void impl_helpMessage(void) {
  printf("   none.\n");
}

void impl_abort(int err) {
  exit(err);
}

/***********************************************************
 * Recursive depth-first implementation                    *
 ***********************************************************/

typedef struct {
  counter_t maxdepth, size, leaves;
} Result;

Result mergeResult(Result r0, Result r1) {
  Result r = {
    (r0.maxdepth > r1.maxdepth) ? r0.maxdepth : r1.maxdepth,
    r0.size + r1.size,
    r0.leaves + r1.leaves
  };
  return r;
}

Node makeChild(const Node *parent, int childType, int computeGranuarity, counter_t idx) {
  int j;

  Node c = { childType, (int)parent->height + 1, -1, {{0}} };

  for (j = 0; j < computeGranularity; j++) {
    rng_spawn(parent->state.state, c.state.state, (int)idx);
  }

  return c;
}

#if UTS_USE_VECTOR

struct dynamic_node {
  int n_children;
  my_ityr::global_vector<global_ptr<dynamic_node>> children;
};

global_ptr<dynamic_node> new_dynamic_node(int n_children) {
  auto gptr = my_ityr::iro::malloc_local<dynamic_node>(1);
  my_ityr::with_checkout_tied<my_ityr::access_mode::write>(
      gptr, 1, [&](auto&& p) {
    new (p) dynamic_node;
    // FIXME: std::launder is not yet implemented in Fujitsu compiler (clang mode)
    /* std::launder(p)->n_children = n_children; */
    /* std::launder(p)->children.resize(n_children); */
    reinterpret_cast<decltype(p)>(p)->n_children = n_children;
    reinterpret_cast<decltype(p)>(p)->children.resize(n_children);
  });
  return gptr;
}

global_ptr<global_ptr<dynamic_node>> get_children(global_ptr<dynamic_node> node) {
  return my_ityr::with_checkout_tied<my_ityr::access_mode::read>(
      node, 1, [&](auto&& p) {
    return p->children.data();
  });
}

void delete_dynamic_node(global_ptr<dynamic_node> node, int) {
  my_ityr::with_checkout_tied<my_ityr::access_mode::read_write>(
      node, 1, [&](auto&& p) {
    std::destroy_at(p);
  });
  my_ityr::iro::free(node, 1);
}

#else

struct dynamic_node {
  int n_children;
  global_ptr<dynamic_node> children[1];
};

std::size_t node_size(int n_children) {
  return sizeof(dynamic_node) + (n_children - 1) * sizeof(global_ptr<dynamic_node>);
}

global_ptr<dynamic_node> new_dynamic_node(int n_children) {
  auto gptr = global_ptr<dynamic_node>(
      my_ityr::iro::malloc_local<std::byte>(node_size(n_children)));
  gptr->*(&dynamic_node::n_children) = n_children;
  return gptr;
}

void delete_dynamic_node(global_ptr<dynamic_node> node, int n_children) {
  my_ityr::iro::free(global_ptr<std::byte>(node), node_size(n_children));
}

global_ptr<global_ptr<dynamic_node>> get_children(global_ptr<dynamic_node> node) {
  return global_ptr<global_ptr<dynamic_node>>(&(node->*(&dynamic_node::children)));
}

#endif

global_ptr<dynamic_node> build_tree(Node parent) {
  counter_t numChildren = uts_numChildren(&parent);
  int childType = uts_childType(&parent);

  global_ptr<dynamic_node> this_node = new_dynamic_node(numChildren);
  global_ptr<global_ptr<dynamic_node>> children = get_children(this_node);

  if (numChildren > 0) {
    my_ityr::parallel_transform(
        ityr::count_iterator<counter_t>(0),
        ityr::count_iterator<counter_t>(numChildren),
        children,
        [=](counter_t i) {
          Node child = makeChild(&parent, childType,
                                 computeGranularity, i);
          return build_tree(child);
        });
  }

  return this_node;
}

Result traverse_tree(counter_t depth, global_ptr<dynamic_node> this_node) {
  counter_t numChildren = this_node->*(&dynamic_node::n_children);
  global_ptr<global_ptr<dynamic_node>> children = get_children(this_node);

  if (numChildren == 0) {
    return { depth, 1, 1 };
  } else {
    Result init { 0, 0, 0 };
    Result result = my_ityr::parallel_reduce(
        children,
        children + numChildren,
        init,
        mergeResult,
        [=](global_ptr<dynamic_node> child_node) {
          return traverse_tree(depth + 1, child_node);
        });
    result.size += 1;
    return result;
  }
}

void destroy_tree(global_ptr<dynamic_node> this_node) {
  counter_t numChildren = this_node->*(&dynamic_node::n_children);
  global_ptr<global_ptr<dynamic_node>> children = get_children(this_node);

  if (numChildren > 0) {
    my_ityr::parallel_for<my_ityr::iro::access_mode::read>(
        children,
        children + numChildren,
        [=](global_ptr<dynamic_node> child_node) {
          destroy_tree(child_node);
        });
  }

  delete_dynamic_node(this_node, numChildren);
}

//-- main ---------------------------------------------------------------------

void uts_run() {
  int my_rank = my_ityr::rank();

  global_ptr<dynamic_node> root_node;

  for (int i = 0; i < numRepeats; i++) {
#if UTS_REBUILD_TREE
    if (my_rank == 0) {
#else
    if (my_rank == 0 && i == 0) {
#endif
      uint64_t t1 = uts_wctime();
      Node root;
      uts_initRoot(&root, type);
      root_node = my_ityr::root_spawn([=]() { return build_tree(root); });
      uint64_t t2 = uts_wctime();

      printf("## Tree built. (%ld ns)\n", t2 - t1);
      fflush(stdout);
    }

    my_ityr::barrier();

    if (my_rank == 0) {
      uint64_t t1 = uts_wctime();
      Result r = my_ityr::root_spawn([=]() { return traverse_tree(0, root_node); });
      uint64_t t2 = uts_wctime();
      uint64_t walltime = t2 - t1;

      counter_t maxTreeDepth = r.maxdepth;
      counter_t nNodes = r.size;
      counter_t nLeaves = r.leaves;

      double perf = (double)nNodes / walltime;

      printf("[%d] %ld ns %.6g Gnodes/s ( nodes: %llu depth: %llu leaves: %llu )\n",
             i, walltime, perf, nNodes, maxTreeDepth, nLeaves);
      fflush(stdout);
    }

    my_ityr::barrier();

#if UTS_REBUILD_TREE
    if (my_rank == 0) {
#else
    if (my_rank == 0 && i == numRepeats - 1) {
#endif
      uint64_t t1 = uts_wctime();
      my_ityr::root_spawn([=]() { destroy_tree(root_node); });
      uint64_t t2 = uts_wctime();

      printf("## Tree destroyed. (%ld ns)\n", t2 - t1);
      fflush(stdout);
    }

    my_ityr::barrier();
    my_ityr::iro::collect_deallocated();
    my_ityr::barrier();
  }
}

void real_main(int argc, char *argv[]) {
  uts_parseParams(argc, argv);

  int my_rank = my_ityr::rank();
  int n_ranks = my_ityr::n_ranks();

  if (my_rank == 0) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[UTS++]\n"
           "# of processes:                %d\n"
           "# of repeats:                  %d\n"
           "PCAS cache size:               %ld MB\n"
           "PCAS sub-block size:           %ld bytes\n"
           "-------------------------------------------------------------\n",
           n_ranks, numRepeats, cache_size, sub_block_size);

    if (type == GEO) {
      printf("t (Tree type):                 Geometric (%d)\n"
             "r (Seed):                      %d\n"
             "b (Branching factor):          %f\n"
             "a (Shape function):            %d\n"
             "d (Depth):                     %d\n"
             "-------------------------------------------------------------\n",
             type, rootId, b_0, shape_fn, gen_mx);
    } else if (type == BIN) {
      printf("t (Tree type):                 Binomial (%d)\n"
             "r (Seed):                      %d\n"
             "b (# of children at root):     %f\n"
             "m (# of children at non-root): %d\n"
             "q (Prob for having children):  %f\n"
             "-------------------------------------------------------------\n",
             type, rootId, b_0, nonLeafBF, nonLeafProb);
    } else {
      assert(0); // TODO:
    }
    printf("uth options:\n");
    madm::uth::print_options(stdout);
    printf("=============================================================\n\n");
    printf("PID of the main worker: %d\n", getpid());
    fflush(stdout);
  }

  my_ityr::iro::init(cache_size * 1024 * 1024, sub_block_size);

  uts_run();

  my_ityr::iro::fini();
}

int main(int argc, char **argv) {
  my_ityr::main(real_main, argc, argv);
  return 0;
}
