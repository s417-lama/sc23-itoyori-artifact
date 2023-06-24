# ityrbench

Itoyori Benchmark Suite (old version)

This repository contains benchmarks with the nested fork-join model and convenient C++ wrapper classes.

A newer version of Itoyori will be available here: https://github.com/itoyori/itoyori

A newer version of Itoyori benchmarks will be available here: https://github.com/itoyori/ityrbench

This repository is only for the reproducibility of experimental results. If you want to run Itoyori, please use the newer version of Itoyori.

## About Itoyori Runtime System

The old version of Itoyori consists of two components:
- Threading layer (called MassiveThreads/DM): https://github.com/s417-lama/massivethreads-dm
- PGAS layer (called PCAS): https://github.com/s417-lama/pcas

These two projects are separately managed, and the C++ header library in this repository (`ityr/` dir) integrates these two.

The newer version of Itoyori (https://github.com/itoyori/itoyori) packs them into a single monolithic repository to facilitate code reuse among them.

## Notices on Performance Reproducibility

Itoyori is build on top of MPI-3 RMA, which is expected to be a portable implementation.
However, the performance might not be portable, because of the implementation quality of MPI-3 RMA operations in each MPI implementation.

What is important for Itoyori's performance is that the MPI RMA operations are *truly one-sided*, which means that one-sided operations do not rely on the progress on the target processes.
This can be achieved by offloading one-sided operations to RDMA, but unfortunately many MPI implementations do not provide truly one-sided RMA operations.
This property is quite important for Itoyori because it offers a highly asynchronous execution model.

From our experiences, the following MPI implementations do not rely on the remote progress for RMA operations:
- Fujitsu MPI for A64FX-based systems (over the Tofu-D Interconnect)
- Open MPI v5.0.x with the UCX osc layer (with the `--mca osc_ucx_acc_single_intrinsic true` option)
    - As it is not a stable release, you may encounter some errors or unstable performance

Please be aware that other MPI implementations can cause a significant performance degradation or even a deadlock.

## Benchmarks

- cilksort
    - Recursively parallel merge sort algorithm
    - Originally included in the Cilk project
- uts
    - Unbalanced Tree Search (UTS) benchmark: https://sourceforge.net/projects/uts-benchmark/
    - Count the total number of tree nodes by traversing an unbalanced tree
    - The tree structure is dynamically unfolded (using hash) and not allocated in the heap memory
- uts++
    - Extended version of the UTS benchmark
    - The tree is managed as a in-memory tree data structure (using global pointers)
    - A similar benchmark is used in [Grappa](https://github.com/uwsampa/grappa) (named UTS-Mem)
- exafmm
    - A library for Fast Multipole Method (FMM)
    - Parallelized by nested fork-join constructs (originally shared-memory only)
    - Original repository is named exafmm-beta: https://github.com/exafmm/exafmm-beta

## How to Run

This benchmark suite uses the [Kochi](https://github.com/s417-lama/kochi) workflow management tool for running experiments on computing clusters.

Kochi v0.0.1: https://github.com/s417-lama/kochi/tree/0.0.1

### Setup

Install Kochi v0.0.1 on your machine:
```sh
pip3 install git+https://github.com/s417-lama/kochi.git@0.0.1
```

In the following, we assume that the same computer (or a login and compute nodes sharing the same file system) is used for job submission and job execution (`local` machine in Kochi).
If you want to run benchmarks on multiple types of machines from the local computer, please see the Kochi tutorial to setup the machine configuration.

Kochi tutorial: https://github.com/s417-lama/kochi-tutorial/tree/0.0.1

If you installed Kochi directly to the compute nodes, please set:
```sh
export KOCHI_DEFAULT_MACHINE=local
```

By default, Kochi manages its workspace at `~/.kochi`.
If the home directory is not accessible or you do not want to mess up the home directory, you can set a different location for the Kochi workspace.

For example, if you want to manage the artifacts in the `playground` directory, you can run:
```sh
mkdir playground
cd playground/
export KOCHI_ROOT=$PWD/.kochi
```

to set `playground/.kochi` as the Kochi workspace.

If you explicitly specify `KOCHI_ROOT`, please remember to set the `KOCHI_ROOT` env val when running Kochi commands later.

Then, please clone this repository:
```sh
git clone https://github.com/s417-lama/ityrbench
cd ityrbench/
```

### Build Dependencies

To build dependencies of the ityrbench, execute:
```sh
kochi install \
  -d massivethreads:release \
  -d massivethreads-dm:release \
  -d pcas:release \
  -d massivelogger:release \
  -d libunwind:v1.6.2 \
  -d backward-cpp:v1.6 \
  -d jemalloc:v5.3.0 \
  -d boost:v1.80.0 \
  -d pcg:master
```

If you encounter an error, please make sure that the dependency build scripts in `kochi.yaml` are vaild and match your system's configuration.
Please modify the build scripts in `kochi.yaml` as needed and retry the build.

### Run Benchmarks Individually

First, you need to submit Kochi jobs to execute the benchmark programs.

To execute the Cilksort benchmark, run:
```sh
kochi enqueue -q test cilksort.yaml
```

This will enqueue the job to the job queue named `test`.

Then, launch a Kochi worker on a compute node:
```sh
kochi work -q test
```

This command will repeatedly pop jobs from the job queue `test` and execute them one by one.
When the job queue becomes empty, the worker will exit immediately (unless the `-b` option is passed).

Each job will build the benchmark program by using the project dependencies and then run the benchmark executable.
`cilksort.yaml` contains how to build and run the benchmark.
The `default_params` field in `cilksort.yaml` lists all parameters and their default values, which can be overwritten by passing commandline options when enqueuing jobs.

To run Cilksort with 1G elements on 2 nodes and 48 cores/node:
```sh
kochi enqueue -q test cilksort.yaml n_input=1000000000 nodes=2 cores=48
```

Similarly, you can run `uts++.yaml` and `exafmm.yaml`.

If you encounter an error with mpirun, modify the mpirun options in `run_common.bash` accordingly.

### Run a Set of Benchmarks

#### Setup

The `batches` field in the job config files (e.g., `cilksort.yaml`) contains a set of benchmarking configurations.
Before explaining how to submit these batch jobs, you need to change some parameters in the job config files.

In the following, we use `cilksort.yaml` as an example config file.
First, depending on how many cores and nodes are available on your system, you need to change the parameter `cores` and `nodes`.

`cores` in the `default_params` field is set as follows (by default, 6 cores/node for the local machine):
```yaml
cores:
  ...
  - value: 6
    machines: [local]
```

You can change this value to the number of cores/node in your system.

The default `nodes` value is 1, but you need to change it for batch jobs.
For example, for the `scale1G` batch job, you can set:
```yaml
scale1G:
  ...
  params:
    nodes:
      - value: [1, 2, 4]
        machines: [local]
  ...
```

if you want to run the jobs on 1, 2, and 4 nodes.

To reduce the execution time:
- Set `default_duplicates` to 1.
- Set `repeats` for each batch job to a smaller number (e.g., 3)

#### Submit and Run

Before submitting your first batch job, you need to initialize the git branch for managing the experimental results by running:
```sh
kochi artifact init ../ityrbench_artifacts
```

This will create a git worktree (`ityrbench_artifacts`) at the parent directory and an orphan branch named `kochi_artifacts`.

Then, you can submit a batch job.

If you want to execute the `scale1G` batch job in `cilksort.yaml`, run:
```sh
kochi batch cilksort.yaml scale1G
```

Then, multiple jobs are enqueued to job queues for each node count (`node_<nodes>`).

To launch a worker for the job queue `node_1`, run the following command on a compute node:
```sh
kochi work -q node_1
```

Similarly, workers for `node_2`, `node_4`, ... should be launched by allocating 2, 4, ... nodes from the system's job manager.

#### Gather Experimental Results and Generate Plots

After the experiments are completed, we gather the experimental results.

To pull the results, you need to move to the artifacts worktree dir and run:
```sh
cd ../ityrbench_artifacts/
kochi artifact sync
```

This will pull the commits including experimental results and merge them into the `kochi_artifacts` branch.

To plot these results, let's merge another branch `kochi_artifacts_base`, which includes plotting scripts, to the current `kochi_artifacts` branch:
```sh
git merge origin/kochi_artifacts_base --allow-unrelated-histories
```

Some Python packages are needed to generate plots:
```sh
pip3 install numpy scipy pandas plotly
```

Let's take a look at the `plot/cilksort/scaling.py` script.

First, you need to make sure that the `machine` variable is correctly set.
If the machine name is `local`, you need to set:
```py
machine = "local"
```

Depending on the batch job configuration, you need to change some variables in the `get_parallel_result_1G()` function:
```py
nodes = [1, 2, 4]      # if `nodes` is set to [1, 2, 4]
duplicates = [0, 1, 2] # if `default_duplicates` is set to 3
```

If you do not execute the `serial` or `scale10G` batch jobs, set `False` to the following variables:
```py
plot_serial = True
plot_10G = True
```

Then, you can get the plot by running:
```py
python3 plot/cilksort/scaling.py
```

The output plot file will be located at `figs/cilksort/scaling_exectime_local.html`, which can be opened by the web browser.
