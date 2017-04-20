# Check CUDA Version
CUDA_DIR = /usr/local/cuda

# 32-bit or 64-bit  
OS_TYPE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")

# Compilers
GCC ?= g++
NVC = $(CUDA_DIR)/bin/nvcc  -ccbin $(GCC)

NVCC_FLAGS   := 
#NVCC_FLAGS   += --keep  
#NVCC_FLAGS   += -cubin
NVCC_FLAGS   += -Xcicc -O0 -Xptxas -O0 
NVCC_FLAGS   += -m${OS_TYPE}

# Directories for Header Files 
NV_INC = -I$(CUDA_DIR)/include/ -I$(CUDA_DIR)/samples/common/inc

# Directories for Libraries
ifeq ($(OS_TYPE), 64)
  NV_LIB = -L$(CUDA_DIR)/lib64
else
  NV_LIB = -L$(CUDA_DIR)/lib
endif

LIB_LINKER = -lcudart

#SMS ?= 30 35 52
SMS = 52

ifeq ($(GENCODE_FLAGS),)
  $(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
endif

#-----------------------------------------------------------------------------#
# Build Targets
#-----------------------------------------------------------------------------#

EXE = pipeline_latency

all: $(EXE)

pipeline_latency: pipeline_latency.cu \
  kernel_uint.cu \
  kernel_int.cu \
  kernel_float.cu \
  kernel_double.cu
	$(NVC) $(NV_INC) $(NV_LIB) $(NVCC_FLAGS) $(GENCODE_FLAGS) -o $@ $< $(LIB_LINKER)

.PHONY: clean
clean:
	rm -rf $(EXE) 
