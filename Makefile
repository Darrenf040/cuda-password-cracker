# Makefile for compiling CUDA programs

# Name of the CUDA source file (without extension)
TARGET = hello

# CUDA compiler
NVCC = nvcc

# Compiler flags (optional)
NVCCFLAGS = -arch=sm_52

# Source and output files
SRC = $(TARGET).cu
OUT = $(TARGET)

# Default target
all: $(OUT)

# Compile CUDA file
$(OUT): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $(OUT) $(SRC)

# Clean up build files
clean:
	rm -f $(OUT)