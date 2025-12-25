# ==========================================
# HPC N-Body Project Makefile
# ==========================================

# --- Directory Structure ---
SRC_DIR = src
BIN_DIR = bin
REP_DIR = vtune_reports

# --- Compilers ---
CXX      = g++
MPICXX   = mpicxx
NVCC     = nvcc

# --- Architecture Flags ---
# OPTION 1: Explicitly target your Compute Node (Best for compiling on Login Node)
# Use this if you are running 'make' on the Login node but running code on Skylake nodes.
ARCH_FLAGS = -march=skylake-avx512 -mtune=skylake-avx512

# OPTION 2: Auto-detect (Best if compiling DIRECTLY on the Compute Node)
# To use this, run: make native
NATIVE_FLAGS = -march=native -mtune=native

# --- Optimization & Debug Flags ---
# Standard high-performance flags
COMMON_FLAGS = -O3 -std=c++17 -Wall
# Debug flags for VTune (Frame pointer required for accurate call graphs)
DEBUG_FLAGS  = -g -fno-omit-frame-pointer

# --- Specific Flags for "Ultra" Version ---
# These are the aggressive flags you requested for the AVX-512 serial code
ULTRA_FLAGS  = $(ARCH_FLAGS) -ffast-math -funroll-loops -finline-functions \
               -fno-trapping-math -fno-math-errno -falign-functions=32 \
               -falign-loops=32 -fno-semantic-interposition \
               -fopt-info-vec-optimized=$(REP_DIR)/vec_report_ultra.txt

# --- Libraries ---
OMP_FLAGS = -fopenmp
CUDA_FLAGS = -O3 -arch=sm_70 -lineinfo # Adjust sm_70 to your GPU (e.g., sm_80 for A100)

# ==========================================
# Targets
# ==========================================

# Default target: builds everything
all: directories serial serial_ultra openmp mpi hybrid cuda

# Create directories if they don't exist
directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(REP_DIR)

# 1. Serial (Standard)
serial: $(SRC_DIR)/nbody_serial.cpp
	$(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_serial

# 2. Serial (ULTRA - The one with intrinsics)
serial_ultra: $(SRC_DIR)/nbody_serial_ultra.cpp
	$(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_serial_ultra

# 3. OpenMP
openmp: $(SRC_DIR)/nbody_openmp.cpp
	$(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_openmp

# 4. MPI
mpi: $(SRC_DIR)/nbody_mpi.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_mpi

# 5. Hybrid (MPI + OpenMP)
hybrid: $(SRC_DIR)/nbody_hybrid.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_hybrid

# 6. CUDA
cuda: $(SRC_DIR)/nbody_cuda.cu
	$(NVCC) $(CUDA_FLAGS) -std=c++17 $< -o $(BIN_DIR)/nbody_cuda

# ==========================================
# Special Targets
# ==========================================

# Target: Clean up binaries
clean:
	rm -f $(BIN_DIR)/* $(REP_DIR)/*.txt

# Target: Compile optimized for the CURRENT machine (e.g., if inside a compute node job)
native: 
	$(MAKE) ARCH_FLAGS="$(NATIVE_FLAGS)" all

# Target: Submit a compilation job to the cluster (Slurm example)
# Usage: make deploy_compile
deploy_compile:
	srun -N 1 -n 1 -p cpu --exclusive $(MAKE) native

# ========================================== Help Target ==========================================
help:
	@echo "======================================================================"
	@echo " HPC N-Body Project Makefile"
	@echo "======================================================================"
	@echo "Usage:"
	@echo " make [target]"
	@echo ""
	@echo "Build Targets:"
	@echo " all : Compile ALL versions (Serial, OpenMP, MPI, Hybrid, CUDA)"
	@echo " serial : Compile Standard Serial version"
	@echo " serial_ultra : Compile Optimized AVX-512 Serial version"
	@echo " openmp : Compile OpenMP version"
	@echo " mpi : Compile MPI version"
	@echo " hybrid : Compile Hybrid (MPI + OpenMP) version"
	@echo " cuda : Compile CUDA version"
	@echo ""
	@echo "Advanced Options:"
	@echo " native : Compile with -march=native (Optimized for CURRENT CPU)"
	@echo " deploy_compile : Submit a compilation job to a compute node (via Slurm)"
	@echo " clean : Remove all binaries and optimization reports"
	@echo " help : Show this help message"
	@echo ""
	@echo "Current Settings:"
	@echo " ARCH_FLAGS : $(ARCH_FLAGS)"
	@echo " CUDA_ARCH : sm_70 (Volta/V100)"
	@echo "======================================================================"
