NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include $(BUILD_ARCH) $(KERNEL_DEFINE)
LD_FLAGS    = -lcudart -lm -L/usr/local/cuda/lib64
OBJ         = main.o support.o pv_model.o panel_data.o

.PHONY: default build naive shared coalesced clean resource-usage

naive:
	$(MAKE) clean
	$(MAKE) KERNEL=naive ARCH=$(ARCH) build

default: build

build: $(EXE)
