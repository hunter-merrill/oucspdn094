#!/bin/bash
# Choose the partition that you will use.
# You can choose: oucspdn_gpu or debug_gpu or gpu 
#
#
#SBATCH --partition=oucspdn_gpu
#
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#
#SBATCH --job-name=P4-1_gpu
#SBATCH --time=00:15:00
#SBATCH --job-name=P4-3
#
#
# --------------------- Change the items below here! ---------------------
#
#SBATCH --mail-user=hunter.d.merrill-1@ou.edu
#SBATCH --mail-type=ALL
#
#SBATCH --chdir=/home/oucspdn094/Project_4_Tests/Autograder/Merrill_Hunter_Project_4/Problem_3/
#SBATCH --output=/home/oucspdn094/Project_4_Tests/Autograder/Merrill_Hunter_Project_4/Problem_3/_P4-1_%J_stdout.txt
#SBATCH --error=/home/oucspdn094/Project_4_Tests/Autograder/Merrill_Hunter_Project_4/Problem_3/_P4-1_%J_stderr.txt

#################################################

module purge
module load CUDA/10.1.243-GCC-8.3.0

# Add commands here
make run