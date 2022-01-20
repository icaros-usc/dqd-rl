#!/bin/bash
# Driver script for running src.analysis.robustness locally. Based on
# run_local.sh -- see there and src/analysis/robustness.py for more info.
#
# Note: the manifest path needs to be the path to a file within the container --
# pay attention if you are using a bind mount which mounts a directory somewhere
# different in the container.
#
# Usage:
#   bash scripts/run_robustness_local.sh MANIFEST SEED NUM_WORKERS

print_header() {
  echo
  echo "------------- $1 -------------"
}

# Prints "=" across an entire line.
print_thick_line() {
  printf "%0.s=" $(seq 1 `tput cols`)
  echo
}

#
# Parse command line flags.
#

MANIFEST="$1"
SEED="$2"
NUM_WORKERS="$3"
if [ -z "${NUM_WORKERS}" ]
then
  echo "Usage: bash scripts/run_robustness_local.sh MANIFEST SEED NUM_WORKERS"
  exit 1
fi

set -u  # Uninitialized vars are error.

#
# Setup env.
#

# Remove MuJoCo locks if needed.
bash scripts/rm_mujoco_lock.sh

#
# Run the experiment.
#

SCHEDULER_FILE=".$(date +'%Y-%m-%d_%H-%M-%S')_scheduler_info.json"
PIDS_TO_KILL=()

print_header "Starting Dask scheduler"
singularity exec --cleanenv container.sif \
  dask-scheduler \
    --scheduler-file $SCHEDULER_FILE &
PIDS_TO_KILL+=("$!")
sleep 2 # Wait for scheduler to start.

print_header "Starting Dask workers"
singularity exec --cleanenv container.sif \
  dask-worker \
    --scheduler-file $SCHEDULER_FILE \
    --nprocs $NUM_WORKERS \
    --nthreads 1 &
PIDS_TO_KILL+=("$!")
sleep 5

print_header "Running script"
echo
print_thick_line
singularity exec --cleanenv --nv --bind ./results:/results container.sif \
  python -m src.analysis.robustness \
    --manifest "$MANIFEST" \
    --address "127.0.0.1:8786" \
    --seed "$SEED"
print_thick_line

#
# Clean Up.
#

print_header "Cleanup"
for pid in ${PIDS_TO_KILL[*]}
do
  kill -9 "${pid}"
done

rm $SCHEDULER_FILE
