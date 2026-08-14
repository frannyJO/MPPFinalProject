ARCH ?=
NVCC = nvcc

COMMON_NVCC_FLAGS = -O3 -I/usr/local/cuda/include
LD_FLAGS = -lcudart -lm -L/usr/local/cuda/lib64

COMMON_OBJ = support.o pv_model.o panel_data.o

NAIVE_ARCH = $(ARCH)
ATOMIC_ARCH = $(if $(ARCH),$(ARCH),-arch=sm_60)

.PHONY: all default naive shared coalesced compute clean resource-usage

all: pv_gpu pv_gpu_shared pv_gpu_coalesced pv_gpu_compute

default: all

naive: pv_gpu
shared: pv_gpu_shared
coalesced: pv_gpu_coalesced
compute: pv_gpu_compute


main_naive.o: main.cu kernel_naive.cu support.h pv_model.h panel_data.h
	$(NVCC) -c -o $@ main.cu $(COMMON_NVCC_FLAGS) $(NAIVE_ARCH)

main_shared.o: main.cu kernel_shared.cu support.h pv_model.h panel_data.h
	$(NVCC) -c -o $@ main.cu $(COMMON_NVCC_FLAGS) $(ATOMIC_ARCH) -DKERNEL_SHARED

main_coalesced.o: main.cu kernel_coalesced.cu support.h pv_model.h panel_data.h
	$(NVCC) -c -o $@ main.cu $(COMMON_NVCC_FLAGS) $(ATOMIC_ARCH) -DKERNEL_COALESCED

main_compute.o: main.cu kernel_compute.cu support.h pv_model.h panel_data.h
	$(NVCC) -c -o $@ main.cu $(COMMON_NVCC_FLAGS) $(ATOMIC_ARCH) -DKERNEL_COMPUTE


support.o: support.cu support.h pv_model.h
	$(NVCC) -c -o $@ support.cu $(COMMON_NVCC_FLAGS) $(ATOMIC_ARCH)

pv_model.o: pv_model.c pv_model.h
	$(NVCC) -c -o $@ pv_model.c $(COMMON_NVCC_FLAGS) $(ATOMIC_ARCH)

panel_data.o: panel_data.c panel_data.h
	$(NVCC) -c -o $@ panel_data.c $(COMMON_NVCC_FLAGS) $(ATOMIC_ARCH)

