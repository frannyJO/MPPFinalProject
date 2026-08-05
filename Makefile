NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -lm -L/usr/local/cuda/lib64
EXE	        = pv_gpu
OBJ	        = main.o support.o pv_model.o panel_data.o

default: $(EXE)
