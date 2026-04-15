#!/bin/bash
#SBATCH --partition=oucspdn_cpu
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#
#SBATCH --time=00:15:00
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=hunter.d.merrill-1@ou.edu
#SBATCH --mail-type=ALL
#
#SBATCH --job-name=hw5prob1_same
#
#SBATCH --chdir=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_1
#SBATCH --output=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_1/_P5-1_diff_%J_stdout.txt
#SBATCH --error=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_1/_P5-1_diff_%J_stderr.txt
#
#################################################

echo "Working directory:"
pwd
echo ""

module load intel/2022.2
module load OpenMPI/4.1.4-GCC-11.3.0

make all

mpirun -n 2 ./pingpong_MPI 1000000 time_1M_same.csv
mpirun -n 2 ./pingpong_MPI 2000000 time_2M_same.csv
mpirun -n 2 ./pingpong_MPI 4000000 time_4M_same.csv
mpirun -n 2 ./pingpong_MPI 8000000 time_8M_same.csv
# add for other sizes

