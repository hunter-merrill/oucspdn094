in=GRCh38_50_rna.fna # baseline input file name, can set like "make run_baseline in=other.fna"

all: make_baseline #make_mapreduce

# duplicate this for other files
make_baseline: compute_median_TF_Exp1_baseline.c
	gcc -fopenmp -g -Wall -o compute_median_TF_Exp1_baseline compute_median_TF_Exp1_baseline.c -lm -std=c99

make_mapreduce: compute_median_TF_Exp2_mapreduce.c
	gcc -fopenmp -g -Wall -o compute_median_TF_Exp2_mapreduce compute_median_TF_Exp2_mapreduce.c -lm -std=c99
	

clean:
	$(RM) compute_median_TF_Exp1_baseline
	$(RM) compute_median_TF_Exp2_mapreduce


# Below are commands to help you run your program easily.
# You will need to create more entries for your different files, such as for critical and locks.
run: run_baseline #run_mapreduce

# duplicate this for other files
run_baseline: make_baseline
	./compute_median_TF_Exp1_baseline $(in) OUTPUT_baseline_1th.csv TIME_baseline_1th.csv 1
	./compute_median_TF_Exp1_baseline $(in) OUTPUT_baseline_2th.csv TIME_baseline_2th.csv 2
	./compute_median_TF_Exp1_baseline $(in) OUTPUT_baseline_4th.csv TIME_baseline_4th.csv 4
	./compute_median_TF_Exp1_baseline $(in) OUTPUT_baseline_8th.csv TIME_baseline_8th.csv 8
	
run_mapreduce: make_mapreduce
	./compute_median_TF_Exp2_mapreduce $(in) OUTPUT_mapreduce_1th.csv TIME_mapreduce_1th.csv 1
	./compute_median_TF_Exp2_mapreduce $(in) OUTPUT_mapreduce_2th.csv TIME_mapreduce_2th.csv 2
	./compute_median_TF_Exp2_mapreduce $(in) OUTPUT_mapreduce_4th.csv TIME_mapreduce_4th.csv 4
	./compute_median_TF_Exp2_mapreduce $(in) OUTPUT_mapreduce_8th.csv TIME_mapreduce_8th.csv 8