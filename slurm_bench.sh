#!/bin/bash
#SBATCH --job-name=bench_server
#SBATCH --output=bench_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=96     
#SBATCH --mem=320g
#SBATCH --mem-bind=local
#SBATCH --sockets-per-node=1  

echo "Job started on $(hostname)"

srun --cpus-per-task=64 --cpu-bind=cores ./server_start.sh &
sleep 10
srun --cpus-per-task=4  --cpu-bind=cores python open_test.py


echo "Test finished. Stopping server..."
kill $SERVER_SRUN_PID