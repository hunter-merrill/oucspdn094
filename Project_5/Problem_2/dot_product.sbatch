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
#SBATCH --job-name=hw5prob2
#
#SBATCH --chdir=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_2
#SBATCH --output=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_2/_P5-2_%J_stdout.txt
#SBATCH --error=/home/oucspdn094/Project_5_Tests/Merrill_Hunter_Project_5/Problem_2/_P5-2_%J_stderr.txt
#
#################################################

# sbatch --ntasks=2 --ntasks-per-node=2 dot_product.sbatch 2
# sbatch --ntasks=4 --ntasks-per-node=4 dot_product.sbatch 4
# sbatch --ntasks=8 --ntasks-per-node=8 dot_product.sbatch 8

echo "Working directory:"
pwd
echo ""

FPATH="/home/oucspdn094/Project_5_Tests/test_data/Vectors/"

module load intel/2022.2
module load OpenMPI/4.1.4-GCC-11.3.0

make all

mpirun -n $1 ./dot_product_MPI 262144 ${FPATH}vec1_2^18.csv ${FPATH}vec2_2^18.csv result_$1_18.csv time_$1_18.csv
mpirun -n $1 ./dot_product_MPI 524288 ${FPATH}vec1_2^19.csv ${FPATH}vec2_2^19.csv result_$1_19.csv time_$1_19.csv
mpirun -n $1 ./dot_product_MPI 1048576 ${FPATH}vec1_2^20.csv ${FPATH}vec2_2^20.csv result_$1_20.csv time_$1_20.csv
# add for other sizes