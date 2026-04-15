#!/bin/bash
#SBATCH --partition=oucspdn_cpu
#
#SBATCH --time=00:15:00
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=hunter.d.merrill-1@ou.edu
#SBATCH --mail-type=ALL
#
#SBATCH --job-name=hw5prob4
#
#SBATCH --chdir=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_4
#SBATCH --output=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_4/_P5-4_%J_stdout.txt
#SBATCH --error=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_4/_P5-4_%J_stderr.txt
#
#################################################

# sbatch --ntasks=2 --ntasks-per-node=2 pi.sbatch 2
# sbatch --ntasks=4 --ntasks-per-node=4 pi.sbatch 4
# sbatch --ntasks=8 --ntasks-per-node=8 pi.sbatch 8
# sbatch --ntasks=16 --ntasks-per-node=16 pi.sbatch 16

echo "Working directory:"
pwd
echo ""

module load intel/2022.2
module load OpenMPI/4.1.4-GCC-11.3.0

make all

mpirun -n $1 ./pi_MPI result_${1}p.csv time_${1}p.csv
# Add more here