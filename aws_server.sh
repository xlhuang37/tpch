#!/bin/bash

# Path to the systemd slice cgroup
CGROUP_PATH="/sys/fs/cgroup/highperf.slice"

# Check if the slice exists
if [ ! -d "$CGROUP_PATH" ]; then
    echo "Error: highperf.slice not found. Is the slice started?"
    exit 1
fi

# Get the Parent PID (the shell that called this script)
SHELL_PID=$PPID

echo "Attaching Shell (PID: $SHELL_PID) to High Performance Group..."

# Move the shell PID into the cgroup
# We use sudo tee because redirection (>) happens as the user, not root
echo $SHELL_PID | sudo tee "$CGROUP_PATH/cgroup.procs" > /dev/null

if [ $? -eq 0 ]; then
    echo "Success! Current shell is now in highperf.slice."
else
    echo "Failed to migrate process."
fi
