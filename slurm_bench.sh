#!/bin/bash
#SBATCH --job-name=bench_server
#SBATCH --output=bench_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=24            
#SBATCH --cpus-per-task=4     
#SBATCH --mem=320g
#SBATCH --mem-bind=local
#SBATCH --sockets-per-node=1    # Excellent for avoiding NUMA latency
#SBATCH --time=12:00:00

echo "Job started on $(hostname)"

# --- STEP 1: Start Server (Backgrounded) ---
# We use srun with --exclusive. Slurm will carve out 64 cores 
# and lock them to this specific step.
# Note: We replaced '--pty bash -l' with the actual script command.

srun --ntasks=1 \
     --cpus-per-task=64 \
     --exclusive \
     --cpu-bind=cores \
     --mem-bind=local \
     --distribution=block:block \
     ./server_start.sh & 

# Capture the Process ID of the srun command (not the python process, but the step launcher)
SERVER_SRUN_PID=$!

# Give the server a few seconds to initialize
sleep 10

# --- STEP 2: Start Client/Test (Foreground) ---
# We use the remaining cores. We do NOT use '&' here because 
# we want the job to stay alive while this runs.

srun --ntasks=1 \
     --cpus-per-task=4 \
     --exclusive \
     --cpu-bind=cores \
     --mem-bind=local \
     --distribution=block:block \
     python open_test.py

# --- STEP 3: Cleanup ---
# Once open_test.py finishes, the script continues here.
# We kill the server step.

echo "Test finished. Stopping server..."
kill $SERVER_SRUN_PID