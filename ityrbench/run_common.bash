#!/bin/bash
[[ -z "${PS1+x}" ]] && set -euo pipefail

MPIEXEC=${MPIEXEC:-mpiexec}

$MPIEXEC --version || true

export MADM_RUN__=1
export MADM_PRINT_ENV=1
export PCAS_PRINT_ENV=1
export ITYR_PRINT_ENV=1

STDOUT_FILE=mpirun_out.txt

case $KOCHI_MACHINE in
  wisteria-o)
    export UTOFU_SWAP_PROTECT=1

    ityr_mpirun() {
      local n_processes=$1
      local n_processes_per_node=$2
      local bind_to=$3

      if [[ $bind_to == none ]]; then
        set_cpu_affinity=0
      else
        set_cpu_affinity=1
      fi

      (
        vcoordfile=$(mktemp)
        if [[ $PJM_ENVIRONMENT == INTERACT ]]; then
          tee_cmd="tee $STDOUT_FILE"
          of_opt=""
          trap "rm -f $vcoordfile" EXIT
        else
          export PLE_MPI_STD_EMPTYFILE=off # do not create empty stdout/err files
          tee_cmd="cat"
          of_opt="-of-proc $STDOUT_FILE"
          trap "rm -f $vcoordfile; compgen -G ${STDOUT_FILE}.* && tail -n +1 \$(ls ${STDOUT_FILE}.* -v) | tee $STDOUT_FILE && rm ${STDOUT_FILE}.*" EXIT
          # trap "rm -f $vcoordfile; compgen -G ${STDOUT_FILE}.* && tail -n +1 \$(ls ${STDOUT_FILE}.* -v) > $STDOUT_FILE && head -100 $STDOUT_FILE" EXIT
        fi
        np=0
        if [[ -z ${PJM_NODE_Y+x} ]]; then
          # 1D
          for x in $(seq 1 $PJM_NODE_X); do
            for i in $(seq 1 $n_processes_per_node); do
              echo "($((x-1)))" >> $vcoordfile
              if (( ++np >= n_processes )); then
                break
              fi
            done
          done
        elif [[ -z ${PJM_NODE_Z+x} ]]; then
          # 2D
          for x in $(seq 1 $PJM_NODE_X); do
            for y in $(seq 1 $PJM_NODE_Y); do
              for i in $(seq 1 $n_processes_per_node); do
                echo "($((x-1)),$((y-1)))" >> $vcoordfile
                if (( ++np >= n_processes )); then
                  break 2
                fi
              done
            done
          done
        else
          # 3D
          for x in $(seq 1 $PJM_NODE_X); do
            for y in $(seq 1 $PJM_NODE_Y); do
              for z in $(seq 1 $PJM_NODE_Z); do
                for i in $(seq 1 $n_processes_per_node); do
                  echo "($((x-1)),$((y-1)),$((z-1)))" >> $vcoordfile
                  if (( ++np >= n_processes )); then
                    break 3
                  fi
                done
              done
            done
          done
        fi
        $MPIEXEC $of_opt -n $n_processes \
          --vcoordfile $vcoordfile \
          --mca plm_ple_cpu_affinity $set_cpu_affinity \
          -- setarch $(uname -m) --addr-no-randomize "${@:4}" | $tee_cmd
      )
    }
    ;;
  local)
    ityr_mpirun() {
      # for ChameleonCloud compute_cascadelake_r_ib
      local n_processes=$1
      local n_processes_per_node=$2
      local bind_to=$3

      export UCX_NET_DEVICES="mlx5_2:1"
      export OMPI_MCA_mca_base_env_list="UCX_NET_DEVICES"
      $MPIEXEC -n $n_processes -N $n_processes_per_node \
        --hostfile $HOME/share/hostfile \
        --bind-to $bind_to \
        --mca btl ^ofi \
        --mca osc_ucx_acc_single_intrinsic true \
        -- setarch $(uname -m) --addr-no-randomize "${@:4}" | tee $STDOUT_FILE
    }
    ;;
esac

run_trace_viewer() {
  if [[ -z ${KOCHI_FORWARD_PORT+x} ]]; then
    echo "Trace viewer cannot be launched without 'kochi interact' command."
    exit 1
  fi
  shopt -s nullglob
  MLOG_VIEWER_ONESHOT=false bokeh serve $KOCHI_INSTALL_PREFIX_MASSIVELOGGER/viewer --port $KOCHI_FORWARD_PORT --allow-websocket-origin \* --session-token-expiration 3600 --args ityr_log_*.ignore pcas_log_*.ignore
}

if [[ ! -z ${KOCHI_INSTALL_PREFIX_PCAS+x} ]]; then
  export PCAS_ENABLE_SHARED_MEMORY=$KOCHI_PARAM_SHARED_MEM
  export PCAS_MAX_DIRTY_CACHE_SIZE=$(bc <<< "$KOCHI_PARAM_MAX_DIRTY * 2^20 / 1")
fi

export MADM_STACK_SIZE=$((4 * 1024 * 1024))

if [[ $KOCHI_PARAM_ALLOCATOR == jemalloc ]]; then
  export LD_PRELOAD=${KOCHI_INSTALL_PREFIX_JEMALLOC}/lib/libjemalloc.so${LD_PRELOAD:+:$LD_PRELOAD}
else
  export LD_PRELOAD=${LD_PRELOAD:+$LD_PRELOAD}
fi

if [[ $KOCHI_PARAM_DEBUGGER == 1 ]] && [[ -z "${PS1+x}" ]]; then
  echo "Use kochi interact to run debugger."
  exit 1
fi
