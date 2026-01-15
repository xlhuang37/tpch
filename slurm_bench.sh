#!/bin/bash
#SBATCH --job-name=bench_server
#SBATCH --output=bench_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=96     
#SBATCH --mem=320g
#SBATCH --mem-bind=local
#SBATCH --sockets-per-node=1  

srun --exclusive --cpus-per-task=64 --cpu-bind=cores --mem-bind=local --distribution=block:block ./server_start.sh &
sleep 10
srun --exclusive --cpus-per-task=4  --cpu-bind=cores --mem-bind=local --distribution=block:blockpython open_test.py

