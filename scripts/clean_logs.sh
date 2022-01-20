# Cleans the logs directory.
#
# Specifically, this script removes all logging directories that have already
# been zipped.
#
# Usage:
#     bash scripts/clean_logs.sh
for logdir in logs/*; do
  if [ -e "${logdir}.zip" ]; then
    rm -vrf $logdir $logdir.zip
  fi
done
