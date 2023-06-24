#!/bin/bash
[[ -z "${PS1+x}" ]] && set -euo pipefail

MPICXX=${MPICXX:-mpicxx}

$MPICXX --version

if [[ ! -z ${KOCHI_INSTALL_PREFIX_PCAS+x} ]]; then
  CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_LOGGER_IMPL=impl_$KOCHI_PARAM_LOGGER"
  CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_DIST_POLICY=$KOCHI_PARAM_DIST_POLICY"
  CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_BLOCK_SIZE=$KOCHI_PARAM_BLOCK_SIZE"

  case $KOCHI_PARAM_CACHE_POLICY in
    serial)            CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_POLICY=ityr_policy_serial" ;;
    nocache)           CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_POLICY=ityr_policy_workfirst -DITYR_IRO_DISABLE_CACHE=1" ;;
    writethrough)      CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_POLICY=ityr_policy_workfirst -DITYR_ENABLE_WRITE_THROUGH=1" ;;
    writeback)         CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_POLICY=ityr_policy_workfirst" ;;
    writeback_lazy)    CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_POLICY=ityr_policy_workfirst_lazy" ;;
    writeback_lazy_wl) CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_POLICY=ityr_policy_workfirst_lazy -DITYR_ENABLE_ACQUIRE_WHITELIST=1" ;;
    getput)            CFLAGS="${CFLAGS:+$CFLAGS} -DITYR_POLICY=ityr_policy_workfirst_lazy -DITYR_IRO_GETPUT=1" ;;
    *)                 echo "Unknown cache policy ($KOCHI_PARAM_CACHE_POLICY)"; exit 1 ;;
  esac
fi
