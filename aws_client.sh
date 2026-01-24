#!/bin/bash

# Path to the systemd slice cgroup
CGROUP_PATH="/sys/fs/cgroup/lowperf.slice"

if [ ! -d "$CGROUP_PATH" ]; then
    echo "Error: lowperf.slice not found. Is the slice started?"
    exit 1
fi

SHELL_PID=$PPID

echo "Attaching Shell (PID: $SHELL_PID) to Low Performance Group..."

echo $SHELL_PID | sudo tee "$CGROUP_PATH/cgroup.procs" > /dev/null

if [ $? -eq 0 ]; then
    echo "Success! Current shell is now in lowperf.slice."
else
    echo "Failed to migrate process."
fi
