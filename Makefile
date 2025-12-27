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
# TARGET: Intel Xeon Platinum 8268 (Cascade Lake)
# COMPILER: GCC 8.2.0 (Too old for 'cascadelake' flag)
# STRATEGY: Use 'skylake-avx512' which is the immediate predecessor and fully compatible.
ARCH_FLAGS = -march=skylake-avx512 -mtune=skylake-avx512

# OPTION 2: Auto-detect (Only use if compiling ON the Compute Node directly)
NATIVE_FLAGS = -march=native -mtune=native

# --- Optimization & Debug Flags ---
COMMON_FLAGS = -O3 -std=c++17 -Wall
DEBUG_FLAGS  = -g -fno-omit-frame-pointer

# --- Specific Flags for "Ultra" Versions ---
# Aggressive optimization for AVX-512 (Serial & OpenMP)
# -funroll-loops: Critical for hiding FMA latency on Xeon 8268
ULTRA_FLAGS  = $(ARCH_FLAGS) -ffast-math -funroll-loops -finline-functions \
               -fno-trapping-math -fno-math-errno -falign-functions=32 \
               -falign-loops=32 -fno-semantic-interposition \
               -ftree-vectorize \
               -fopt-info-vec-optimized=$(REP_DIR)/vec_report_ultra.txt

# --- Libraries ---
OMP_FLAGS  = -fopenmp
# Compute Cap 7.0 is correct for Tesla V100 (Volta)
CUDA_FLAGS = -O3 -arch=sm_70 -lineinfo 

# ==========================================
# Targets
# ==========================================

all: directories serial serial_ultra openmp openmp_ultra mpi hybrid cuda

directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(REP_DIR)

# 1. Serial
serial: $(SRC_DIR)/nbody_serial.cpp
	$(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_serial

# 2. Serial Ultra (AVX-512 Intrinsics)
serial_ultra: $(SRC_DIR)/nbody_serial_ultra.cpp
	$(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_serial_ultra

# 3. OpenMP
openmp: $(SRC_DIR)/nbody_openmp.cpp
	$(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_openmp

# 4. OpenMP Ultra (AVX-512 Intrinsics + Optimized)
openmp_ultra: $(SRC_DIR)/nbody_openmp_ultra.cpp
	$(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_openmp_ultra

# 5. MPI
mpi: $(SRC_DIR)/nbody_mpi.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_mpi

# 6. MPI Ultra
mpi_ultra: $(SRC_DIR)/nbody_mpi_ultra.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_mpi_ultra

# 7. Hybrid
hybrid: $(SRC_DIR)/nbody_hybrid.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_hybrid

# 8. Hybrid Ultra
hybrid_ultra: $(SRC_DIR)/nbody_hybrid_ultra.cpp
	$(MPICXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_hybrid_ultra

# 9. CUDA
cuda: $(SRC_DIR)/nbody_cuda.cu
	$(NVCC) $(CUDA_FLAGS) -std=c++17 $< -o $(BIN_DIR)/nbody_cuda

# ==========================================
# Utility Targets
# ==========================================

clean:
	rm -f $(BIN_DIR)/* $(REP_DIR)/*.txt

# Submit compilation to compute node (useful if Login node architecture differs too much)
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
