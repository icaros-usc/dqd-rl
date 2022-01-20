#!/bin/bash
# Dashboard for monitoring experiments in the slurm_logs/ directory.
#
# Usage:
#   scripts/slurm_dashboard.sh

print_header() {
  echo "------------- $1 -------------"
  echo
}

print_header "SLURM LOGS"
i=0
for x in $(ls slurm_logs/); do
  d="slurm_logs/$x"
  i=$(($i + 1))
  sched="$d/scheduler.out"
  main_logdir=$(cat "$d/logdir")
  name=$(grep "experiment.name" "$main_logdir/config.gin" | sed "s/.*'\\(.*\\)'/\\1/g")
  seed=$(cat "$main_logdir/seed")

  echo "$i. $d ($name - Seed $seed)"
  echo "Logdir: $main_logdir"
  echo "Generations: $(grep "\-\-\- Generation" "$sched" | tail -n 1 | sed 's/.* Generation \|\-*//g')"
  echo "Tail: $(tail -n 1 "$sched")"

  echo
done

print_header "SLURM JOBS"
squeue -o '  %10i %.9P %.2t %.8p %.4D %.3C %.10M %30j %R' -u $USER
