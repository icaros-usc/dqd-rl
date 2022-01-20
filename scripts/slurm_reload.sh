#!/bin/bash
# Reloads an experiment. Useful when experiments fail or need to be extended (in
# such cases, modify config.gin in the logging directory before running this
# script).
#
# Usage:
#   bash scripts/slurm_reload.sh [LOGDIR]
#
# Example:
#   bash scripts/slurm_reload.sh logs/.../
LOGDIR="$1"
CONFIG="$LOGDIR/config.gin"
SEED=$(cat "$LOGDIR/seed")
HPC_CONFIG="$LOGDIR/hpc_config.sh"

bash scripts/run_slurm.sh "$CONFIG" "$SEED" "$HPC_CONFIG" -r "$LOGDIR"
