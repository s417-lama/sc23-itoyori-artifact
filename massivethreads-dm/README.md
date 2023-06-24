# MassiveThreads/DM

Forked from: https://github.com/shigeki-akiyama/massivethreads-dm

This repository contains a threading layer of the old version of the Itoyori runtime system.

A newer version of Itoyori will be available here: https://github.com/itoyori/itoyori

This repository is only for the reproducibility of experimental results. If you want to run Itoyori, please use the newer version of Itoyori.

## How to Run Locally

Compile:
```
./configure
make
```

Run a benchmark program (Binary Task Creation; bin):
```
./misc/madmrun/madmrun -n <number of processes> uth/examples/bin/bin <depth> <leaf_loops> <interm_loops> <interm_iters pre_exec>
```

Parameters:
- `depth`: depth of the binary tree
- `leaf_loops`: how many clocks are consumed in leaf tasks (in clocks)
- `interm_loops`: how many clocks are consumed in intermediate tasks in the tree (in clocks)
- `interm_iters`: number of fork-join blocks in intermediate tasks
- `pre_exec`: number of warm-up runs

Example result:
```
$ ./misc/madmrun/madmrun -n 6 uth/examples/bin/bin 7 10000 1000 3 1
program = bin,
depth = 7, leaf_loops = 10000, interm_loops = 1000, interm_iters = 3, pre_exec = 1
MADM_DEBUG_LEVEL = 0, MADM_DEBUG_LEVEL_RT = 5, MADM_CORES = 2147483647, MADM_SERVER_MOD = 0, MADM_GASNET_POLL_THREAD = 0, MADM_GASNET_SEGMENT_SIZE = 0
MADM_STACK_SIZE = 1048576, MADM_TASKQ_CAPACITY = 1024, MADM_PROFILE = 0, MADM_STEAL_LOG = 0, MADM_ABORTING_STEAL = 1
np = 6, server_mod = 0, time = 0.230996,
throughput = 1.454238, throughput/np = 0.242373, task overhead = 4126
```

You may need to modify the option for setting an environment variable in `mpirun` in the script `misc/madmrun/madmrun`:

For OpenMPI:
```
-x MADM_RUN__=1
```

For MPICH:
```
-env MADM_RUN__ 1
```

## Publications

- Shigeki Akiyama and Kenjiro Taura. "Uni-Address Threads: Scalable Thread Management for RDMA-Based Work Stealing." in HPDC '15. https://doi.org/10.1145/2749246.2749272
- Shigeki Akiyama and Kenjiro Taura. "Scalable Work Stealing of Native Threads on an x86-64 Infiniband Cluster." Journal of Information Processing (JIP). 2016. https://doi.org/10.2197/ipsjjip.24.583
- Shumpei Shiina and Kenjiro Taura. "Distributed Continuation Stealing is More Scalable than You Might Think." in Cluster '22. https://doi.org/10.1109/CLUSTER51413.2022.00027
