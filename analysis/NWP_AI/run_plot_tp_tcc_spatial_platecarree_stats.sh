#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENMP_STUBS="/home/lhwang/anaconda3/pkgs/intel-openmp-2023.1.0-hdb19cb5_46306/lib/libiompstubs5.so"

export LD_PRELOAD="${OPENMP_STUBS}${LD_PRELOAD:+:${LD_PRELOAD}}"
python "${SCRIPT_DIR}/plot_tp_tcc_spatial_platecarree_stats.py"
