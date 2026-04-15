NVCC        = nvcc
GCC         = gcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64

# Executables
EXE_P1      = gpu_mining_problem1
EXE_STARTER = gpu_mining_starter
EXE_SERIAL  = serial_mining

# Object files
OBJ_P1      = gpu_mining_problem1.o support.o
OBJ_STARTER = gpu_mining_starter.o support.o

# Default target now builds Problem 1
default: $(EXE_P1)

# Builds all three executables
all: $(EXE_P1) $(EXE_STARTER) $(EXE_SERIAL)

# --- Executable Compilation Rules ---

$(EXE_P1): $(OBJ_P1)
	$(NVCC) -o $@ $^ $(LD_FLAGS)

$(EXE_STARTER): $(OBJ_STARTER)
	$(NVCC) -o $@ $^ $(LD_FLAGS)

$(EXE_SERIAL): serial_mining.c
	$(GCC) -O3 -fopenmp -o $@ $^

# ------------------------------------------------------------------------------- #

# input file and size
FILE     = /home/oucspdn094/Project_4_Tests/Autograder/test_data/Problem_1_and_2/in_20k.csv
SIZE	 = 20000

# number of trials to run
TRIALS_A = 5000000
TRIALS_B = 10000000

# output file suffix
OUT_A    = 20k_5m
OUT_B    = 20k_10m

# --- Run Commands ---

run: $(EXE_P1)
	clear 
	./$(EXE_P1) $(FILE) $(SIZE) $(TRIALS_A) p1_out_$(OUT_A).csv p1_time_$(OUT_A).csv
	./$(EXE_P1) $(FILE) $(SIZE) $(TRIALS_B) p1_out_$(OUT_B).csv p1_time_$(OUT_B).csv

run_all: all
	clear
	./$(EXE_SERIAL) $(FILE) $(SIZE) $(TRIALS_A) serial_out_$(OUT_A).csv serial_time_$(OUT_A).csv
	./$(EXE_SERIAL) $(FILE) $(SIZE) $(TRIALS_B) serial_out_$(OUT_B).csv serial_time_$(OUT_B).csv
	
	./$(EXE_STARTER) $(FILE) $(SIZE) $(TRIALS_A) starter_out_$(OUT_A).csv starter_time_$(OUT_A).csv
	./$(EXE_STARTER) $(FILE) $(SIZE) $(TRIALS_B) starter_out_$(OUT_B).csv starter_time_$(OUT_B).csv
	
	./$(EXE_P1) $(FILE) $(SIZE) $(TRIALS_A) p1_out_$(OUT_A).csv p1_time_$(OUT_A).csv
	./$(EXE_P1) $(FILE) $(SIZE) $(TRIALS_B) p1_out_$(OUT_B).csv p1_time_$(OUT_B).csv

# ------------------------------------------------------------------------------- #

# --- Object Compilation Rules ---

gpu_mining_problem1.o: gpu_mining_problem1.cu nonce_kernel.cu hash_kernel.cu support.h
	$(NVCC) -c -o $@ gpu_mining_problem1.cu $(NVCC_FLAGS) $(LD_FLAGS)

gpu_mining_starter.o: gpu_mining_starter.cu nonce_kernel.cu support.h
	$(NVCC) -c -o $@ gpu_mining_starter.cu $(NVCC_FLAGS) $(LD_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE_P1) $(EXE_STARTER) $(EXE_SERIAL)