#!/bin/bash
# Runs experiments on a local computer.
#
# Usage:
#   bash scripts/run_local.sh CONFIG SEED NUM_WORKERS [RELOAD_PATH]
# Example:
#   # 8 workers with configuration config/foo.gin and seed 1.
#   bash scripts/run_local.sh config/foo.gin 1 8
#
#   # 4 workers with configuration config/foo.gin and seed 2, and reloading from
#   # old_dir/.
#   bash scripts/run_local.sh config/foo.gin 2 4 old_dir/

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

CONFIG="$1"
SEED="$2"
NUM_WORKERS="$3"
RELOAD_PATH="$4"
if [ -z "${NUM_WORKERS}" ]
then
  echo "Usage: bash scripts/run_local.sh CONFIG SEED NUM_WORKERS [RELOAD_PATH]"
  exit 1
fi

if [ -n "$RELOAD_PATH" ]; then
  RELOAD_ARG="--reload-dir ${RELOAD_PATH}"
else
  RELOAD_ARG=""
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

print_header "Running experiment"
echo
print_thick_line
singularity exec --cleanenv --nv container.sif \
  python -m src.main \
    --config "$CONFIG" \
    --address "127.0.0.1:8786" \
    --seed "$SEED" \
    $RELOAD_ARG
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
