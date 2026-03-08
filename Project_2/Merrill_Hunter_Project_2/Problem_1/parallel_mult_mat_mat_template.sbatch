#!/bin/bash
# Choose the queue that you want to use
#SBATCH --partition=oucspdn_cpu
#SBATCH --exclusive
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --chdir=/home/oucspdn094/Project_2/Problem_1
#SBATCH --output=/home/oucspdn094/Project_2/Problem_1/Problem1_%J_stdout.txt
#SBATCH --error=/home/oucspdn094/Project_2/Problem_1/Problem1_%J_stderr.txt
# Adjust time according to your needs and queue limit 
#SBATCH --time=00:15:00
#SBATCH --job-name=Project2_Problem1
#SBATCH --mail-user=hunter.d.merrill-1@ou.edu
#SBATCH --mail-type=ALL

#################################################

echo "**Test 1 - 1 threads**"
./parallel_mult_mat_mat test1_input_mat_a.csv 1000 1000 test1_input_mat_b.csv 1000 1000 output1.csv test1_time_1p.csv 1

echo "**Test 2 - 2 threads**"
./parallel_mult_mat_mat test2_input_mat_a.csv 1000 1000 test2_input_mat_b.csv 1000 2000 output2.csv test2_time_2p.csv 2

echo "**Test 3 - 4 threads**"
./parallel_mult_mat_mat test3_input_mat_a.csv 2000 1000 test3_input_mat_b.csv 1000 2000 output3.csv test3_time_3p.csv 4

echo "**Test 4 - 8 threads**"
./parallel_mult_mat_mat test4_input_mat_a.csv 2000 2000 test4_input_mat_b.csv 2000 2000 output4.csv test4_time_4p.csv 8