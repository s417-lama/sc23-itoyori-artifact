# Artifacts of the paper "Itoyori: Reconciling Global Address Space with Global Fork-Join Task Parallelism" at SC '23

This repository is tailored to execution on [ChameleonCloud](https://www.chameleoncloud.org).
Please consult [s417-lama/ityrbench](https://github.com/s417-lama/ityrbench) for more general information (e.g., running with a batch job system on supercomputers).

Note: A newer version of Itoyori is available here: https://github.com/itoyori/itoyori
We recommend to use the newer version of Itoyori for other purposes than reproducing our experiments.

## Setup Environment

### Node Setup

First, we have to make a reservation to allocate several nodes with the following filters:
- `node_type == compute_cascadelake_r`
- `infiniband == True`

After the nodes are allocated, we launch an instance to setup InfiniBand and Open MPI.
The base image we chose is `CC-Ubuntu22.04`.

The following setup in this section is already saved as an image named `itoyori-base`.
Here, we explain a step-by-step instruction to reproduce this image.

First, upgrade the Linux version to the latest:
```sh
sudo apt update
sudo apt install --upgrade -y linux-image-generic
sudo reboot
```

After the reboot, install InfiniBand driver (MLNX_OFED-23.04-1.1.3.0):
```sh
wget https://content.mellanox.com/ofed/MLNX_OFED-23.04-1.1.3.0/MLNX_OFED_LINUX-23.04-1.1.3.0-ubuntu22.04-x86_64.iso
sudo mount -o ro,loop MLNX_OFED_LINUX-23.04-1.1.3.0-ubuntu22.04-x86_64.iso /mnt
sudo /mnt/mlnxofedinstall --force
sudo /etc/init.d/openibd restart
```

After installing the InfiniBand driver, please check that the port state is `Active` and its physical state is `LinkUp` by executing `ibstat` command:
```console
$ ibstat
...
CA 'mlx5_2'
        CA type: MT4123
        Number of ports: 1
        Firmware version: 20.28.4512
        Hardware version: 0
        Node GUID: 0x1c34da03004743c0
        System image GUID: 0x1c34da03004743c0
        Port 1:
                State: Active
                Physical state: LinkUp
                Rate: 100
                Base lid: 5
                LMC: 0
                SM lid: 1
                Capability mask: 0x2651e848
                Port GUID: 0x1c34da03004743c0
                Link layer: InfiniBand
```

Check UCX installation:
```console
$ ucx_info -v
# Library version: 1.15.0
# Library path: /lib/libucs.so.0
# API headers version: 1.15.0
# Git branch '', revision d799cfd
# Configured with: --disable-logging --disable-debug --disable-assertions --disable-params-check --enable-mt --prefix=/usr --enable-examples --with-java=no --with-cuda=/usr/local/cuda --without-xpmem
```

Then, we manually install Open MPI v5.0.0rc11.
We prefer v5.0.x because of the support for the `osc_ucx_acc_single_intrinsic` MCA option, which allows for RMA atomic operations to be directly mapped to the network atomic operations.

Install Open MPI v5.0.0rc11:
```sh
git clone https://github.com/open-mpi/ompi.git
cd ompi/
git checkout v5.0.0rc11
git submodule update --init --recursive
./autogen.pl
mkdir build
cd build/
../configure --disable-man-pages --with-ucx --with-pmix=internal --with-hwloc=internal --with-libevent=internal
make -j
sudo make install
sudo ldconfig
```

Check Open MPI installation:
```console
$ mpirun --version
mpirun (Open MPI) 5.0.0rc11

Report bugs to https://www.open-mpi.org/community/help/
```

Finally, install the following utilities needed to build our artifacts:
```sh
sudo apt install numactl cmake binutils-dev
```

The above settings were saved as an image `itoyori-base`:
```sh
sudo cc-snapshot itoyori-base
```

Then, the `itoyori-base` image can be deployed to all nodes allocated as a lease.

### Setup a Shared File System

A shared file system is needed to run our artifacts.

Here, we will setup a shared file system by our own.
Although ChameleonCloud provides a way to configure a shared file system (called a *share*), it caused an error in our trial, so we do not use it in this instruction.

First, we launch a NFS server on one of the nodes (we call this node the *main node*):
```sh
sudo apt install nfs-kernel-server
mkdir ~/share
echo '/home/cc/share 10.31.0.176/28(rw,sync,no_subtree_check,no_root_squash)' | sudo tee -a /etc/exports
sudo systemctl restart nfs-server
```

Please replace `10.31.0.176/28` with the subnet of your network, which covers all nodes you allocated.

Then, on the remaining nodes (called *sub nodes*), we connect to the NFS:
```sh
mkdir ~/share
sudo mount -t nfs -o nfsvers=4.2,proto=tcp 10.31.0.181:/home/cc/share share
```

Please replace `10.31.0.181` with the IP address of the main node.
This setup should be done for every sub node.

### Run Open MPI

Then, let's check if Open MPI works on multiple nodes.

First, create a hostfile at `~/share/hostfile`:
```
<IP address of node 1>:48
<IP address of node 2>:48
<IP address of node 3>:48
...
<IP address of node N>:48
```

`48` is the number of physical cores on a `compute_cascadelake_r` node.

After creating a hostfile, run mpirun on the main node:
```sh
mpirun --hostfile ~/share/hostfile -n <N> -N 1 hostname
```

It is a success if all hostnames are shown in the console.
If the `mpirun` command gets stuck, please login to other sub nodes with ssh (to add them to *known hosts*) and try again.

### Setup This Artifact Using Kochi

First, please download and expand our repository to under `~/share`.
We will assume that this repository is placed at `~/share/sc23-itoyori-artifact`.

You need to first install [Kochi](https://github.com/s417-lama/kochi) workflow management tool, which is also included in this repository (`kochi/` dir).

To install:
```sh
cd ~/share/sc23-itoyori-artifact/kochi/
pip3 install . --upgrade
```

Then, add the following configurations to `~/.bashrc`:
```sh
echo 'export KOCHI_ROOT=$HOME/share/.kochi' >> ~/.bashrc
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
echo 'export KOCHI_DEFAULT_MACHINE=local' >> ~/.bashrc
```

Then, reload `~/.bashrc`.

Note that the Kochi's root directory (`~/share/.kochi`) is configurable, but it should be placed under the shared file system.
Changing `KOCHI_DEFAULT_MACHINE` is useful when we have to use some batch job system, but in a baremetal cloud like ChameleonCloud, we don't need to care about it.
Please see [Kochi](https://github.com/s417-lama/kochi) and [Kochi tutorial](https://github.com/s417-lama/kochi-tutorial) if you are interested.

Kochi requires the benchmark project be managed as a git repository.
To initialize the git repository at the `ityrbench/` dir:
```sh
cd ~/share/sc23-itoyori-artifact/ityrbench
git init
git add .
git config user.name "test"
git config user.email "<>"
git commit -m "git init"
```

Then, build and install the dependencies for the benchmark suite (ityrbench):
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

You can test Itoyori under the `ityrbench/` dir as follows:
```sh
kochi enqueue -q test cilksort.yaml n_input=1000000000 nodes=1
kochi enqueue -q test cilksort.yaml n_input=1000000000 nodes=2
kochi work -q test
```

The above command enqueues two jobs with different node counts (1 and 2) to a job queue `test`, and then creates a Kochi worker to work on these jobs.

## Run Experiments

After the above setup is done, let's run experiments included in the ityrbench dir.
The following commands are expected to be executed at `~/share/sc23-itoyori-artifact/ityrbench`.

### General Instruction

Kochi has three components: jobs, job queues, and workers.
A job specifies how to build and run each benchmark, which is enqueued to a job queue.
A worker is a process which executes each job by repeatedly popping jobs from a specified job queue.
The user first submits jobs to job queues and then launch workers on compute nodes allocated by the system's job manager.
The user can also specify a set of jobs with different parameters (e.g., the number of nodes, the input size), which is called a batch job.
To reproduce the figures in this paper, we submit batch jobs in each job configuration file (`cilksort.yaml`, `uts++.yaml`, and `exafmm.yaml`).

Before submitting the first batch job, we have to create another git branch to save the output by each job:
```sh
kochi artifact init ../ityrbench_artifacts
```

To submit a batch job, we run:
```sh
kochi batch <job_config_file> <batch_name>
```

Depending on the batch configuration, several jobs are submitted to multiple job queues, named `node_<N>`, where `<N>` is the number of nodes required to run that job.

We can launch a Kochi worker for each number of nodes:
```sh
kochi work -q node_<N>
```

Note that the job queues are separated for each node count to be incorporated with batch job management systems.

After the jobs are completed, we have to collect the output from the jobs, by running:
```sh
kochi artifact sync
```

Then, the output will appear at the `../ityrbench_artifacts` dir.

Visualization for these output files is performed by Python scripts under `plot_scripts`.
The required packages for plotting are:
```sh
pip3 install numpy scipy pandas plotly
```

The plot output will be saved at `figs/` dir, and these HTML files can be viewed using a web browser.

### Cilksort: Granularity Plot (Figure 7)

The default setting in this repository is to run on 1, 2, and 4 nodes.
Please configure the `nodes` param in each experiment file (`*.yaml`).

Run experiments:
```sh
kochi batch cilksort.yaml granularity # about 20 min
kochi work -q node_4
```

Plot:
```sh
python3 ../plot_scripts/cilksort/granularity.py
```

Please remember to run `kochi artifact sync` before plotting.

### Cilksort: Scaling Plot (Figure 8)

Run experiments with 1G elems:
```sh
kochi batch cilksort.yaml scale1G # about 5 min
kochi work -q node_1 && kochi work -q node_2 && kochi work -q node_4
```

(Optional) Serial execution:
```sh
kochi batch cilksort.yaml serial # about 20 min
kochi work -q node_1
```

(Optional) experiments with 10G elems (takes very long time):
```sh
kochi batch cilksort.yaml scale10G
kochi work -q node_1 && kochi work -q node_2 && kochi work -q node_4
```

Plot:
```sh
python3 ../plot_scripts/cilksort/scaling.py
```

### Cilksort: Statistics (Figure 9)

The above experiment already collected required data.

Plot:
```sh
python3 ../plot_scripts/cilksort/scaling.py
```

### UTS-Mem: Scaling Plot (Figure 10)

Run experiments with T1L tree:
```sh
kochi batch uts++.yaml T1L # about 5 min
kochi work -q node_1 && kochi work -q node_2 && kochi work -q node_4
```

(Optional) Serial execution:
```sh
kochi batch uts++.yaml serial # about 3 min
kochi work -q node_1
```

Plot:
```sh
python3 ../plot_scripts/uts++/scaling.py
```

### ExaFMM: Scaling Plot (Figure 11)

Run experiments with 1M elems:
```sh
kochi batch exafmm.yaml scale1M # about 15 min
kochi batch exafmm_mpi.yaml scale1M # about 2 min
kochi work -q node_1 && kochi work -q node_2 && kochi work -q node_4
```

(Optional) Serial execution:
```sh
kochi batch exafmm.yaml serial
kochi work -q node_1
```

(Optional) experiments with 10M elems (takes very long time):
```sh
kochi batch exafmm.yaml scale10M
kochi batch exafmm_mpi.yaml scale10M
kochi work -q node_1 && kochi work -q node_2 && kochi work -q node_4
```

Plot:
```sh
python3 ../plot_scripts/exafmm/scaling.py
```

### Configure Experimental Settings

To reduce the time to reproduce the experiments, we reduced the number of iterations (`repeats` param) and the number of duplicated job submissions (`default_duplicates`) from our original experiment.
You can configure them by modifying `*.yaml` files.
The original value of `repeats` was 11 and that of `default_duplicates` was 3.
