
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = convolution_maxpooling_CUDA
OBJ	        = convolution_maxpooling_CUDA.o support.o

default: $(EXE)

# ------------------------------------------------------------------------------- #

# input file and size
IN       = /home/oucspdn094/Project_4_Tests/Autograder/test_data/Problem_4/mat_input.csv
OUT      = /home/oucspdn094/Project_4_Tests/Autograder/Merrill_Hunter_Project_4/Problem_4/mat_output_prob4.csv
TIME     = /home/oucspdn094/Project_4_Tests/Autograder/Merrill_Hunter_Project_4/Problem_4/time_prob4_CUDA.csv

N_ROW	 = 2048
N_COL	 = 2048

run:
	clear 
	make
	./convolution_maxpooling_CUDA $(N_ROW) $(N_COL) $(IN) $(OUT) $(TIME)

# ------------------------------------------------------------------------------- #

convolution_maxpooling_CUDA.o: convolution_maxpooling_CUDA.cu kernel.cu support.h
	$(NVCC) -c -o $@ convolution_maxpooling_CUDA.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
