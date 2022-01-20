#!/bin/bash
# Removes MuJoCo locks if needed. Intended to be used when building the
# Singularity container (see container.def).

MUJOCO_LOCKS=("$HOME/mujocopy-buildlock" "/tmp/mujocopy-buildlock")

for lock in "${MUJOCO_LOCKS[@]}"
do
  if [ -e "$lock" ]
  then
    echo "Removing $lock"
    rm $lock
  fi
done
