SRC_DIR = src
BIN_DIR = bin
REP_DIR = vtune_reports

CPU_ENV = . /home/apps/spack/share/spack/setup-env.sh && \
          spack load gcc@13.1.0%gcc@13.1.0 && \
          module load openmpi/4.1.1 && \
          module load oneapi/vtune/2021.7.1

GPU_ENV = module purge && \
          . /home/apps/spack/share/spack/setup-env.sh && \
          spack load gcc@11.2.0 && \
          module load cuda/12.0

CXX      = g++
MPICXX   = mpicxx
NVCC     = nvcc

ARCH_FLAGS = -march=cascadelake -mtune=cascadelake
COMMON_FLAGS = -O3 -std=c++17 -Wall
DEBUG_FLAGS  = -g -fno-omit-frame-pointer

ULTRA_FLAGS  = $(ARCH_FLAGS) -ffast-math -funroll-loops -finline-functions \
               -fno-trapping-math -fno-math-errno -falign-functions=32 \
               -falign-loops=32 -fno-semantic-interposition \
               -ftree-vectorize -mprefer-vector-width=512 \
               -fipa-pta \
               -fopt-info-vec-optimized=$(REP_DIR)/vec_report_ultra.txt

OMP_FLAGS  = -fopenmp
CUDA_FLAGS = -O3 -arch=sm_70 -lineinfo

all: directories batch_cpu batch_gpu

directories:
	@mkdir -p $(BIN_DIR)
	@mkdir -p $(REP_DIR)

cpu:
	@echo "Building ALL CPU Codes..."
	@$(CPU_ENV) && \
	$(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $(SRC_DIR)/nbody_serial.cpp -o $(BIN_DIR)/nbody_serial && \
	$(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $(SRC_DIR)/nbody_serial_ultra.cpp -o $(BIN_DIR)/nbody_serial_ultra && \
	$(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $(SRC_DIR)/nbody_openmp.cpp -o $(BIN_DIR)/nbody_openmp && \
	$(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $(SRC_DIR)/nbody_openmp_ultra.cpp -o $(BIN_DIR)/nbody_openmp_ultra && \
	$(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $(SRC_DIR)/nbody_mpi.cpp -o $(BIN_DIR)/nbody_mpi && \
	$(MPICXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $(SRC_DIR)/nbody_mpi_ultra.cpp -o $(BIN_DIR)/nbody_mpi_ultra && \
	$(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $(SRC_DIR)/nbody_hybrid.cpp -o $(BIN_DIR)/nbody_hybrid && \
	$(MPICXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $(SRC_DIR)/nbody_hybrid_ultra.cpp -o $(BIN_DIR)/nbody_hybrid_ultra

gpu:
	@echo "Building ALL GPU Codes..."
	@$(GPU_ENV) && \
	$(NVCC) $(CUDA_FLAGS) -std=c++17 $(SRC_DIR)/nbody_cuda.cu -o $(BIN_DIR)/nbody_cuda && \
	$(NVCC) $(CUDA_FLAGS) --use_fast_math -std=c++17 $(SRC_DIR)/nbody_cuda_ultra.cu -o $(BIN_DIR)/nbody_cuda_ultra

serial: $(SRC_DIR)/nbody_serial.cpp
	@$(CPU_ENV) && $(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_serial

serial_ultra: $(SRC_DIR)/nbody_serial_ultra.cpp
	@$(CPU_ENV) && $(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_serial_ultra

openmp: $(SRC_DIR)/nbody_openmp.cpp
	@$(CPU_ENV) && $(CXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_openmp

openmp_ultra: $(SRC_DIR)/nbody_openmp_ultra.cpp
	@$(CPU_ENV) && $(CXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_openmp_ultra

mpi: $(SRC_DIR)/nbody_mpi.cpp
	@$(CPU_ENV) && $(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_mpi

mpi_ultra: $(SRC_DIR)/nbody_mpi_ultra.cpp
	@$(CPU_ENV) && $(MPICXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_mpi_ultra

hybrid: $(SRC_DIR)/nbody_hybrid.cpp
	@$(CPU_ENV) && $(MPICXX) $(COMMON_FLAGS) $(ARCH_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_hybrid

hybrid_ultra: $(SRC_DIR)/nbody_hybrid_ultra.cpp
	@$(CPU_ENV) && $(MPICXX) $(COMMON_FLAGS) $(ULTRA_FLAGS) $(OMP_FLAGS) $(DEBUG_FLAGS) $< -o $(BIN_DIR)/nbody_hybrid_ultra

cuda: $(SRC_DIR)/nbody_cuda.cu
	@$(GPU_ENV) && $(NVCC) $(CUDA_FLAGS) -std=c++17 $< -o $(BIN_DIR)/nbody_cuda

cuda_ultra: $(SRC_DIR)/nbody_cuda_ultra.cu
	@$(GPU_ENV) && $(NVCC) $(CUDA_FLAGS) --use_fast_math -std=c++17 $< -o $(BIN_DIR)/nbody_cuda_ultra

check_env:
	@echo "--- Checking CPU Environment ---"
	@$(CPU_ENV) && gcc --version && echo "---" && mpicxx --version
	@echo " "
	@echo "--- Checking GPU Environment ---"
	@$(GPU_ENV) && gcc --version && echo "---" && nvcc --version

clean:
	rm -f $(BIN_DIR)/* $(REP_DIR)/*.txt

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  all           : Fast batch compile of EVERYTHING"
	@echo "  cpu     : Fast batch compile of CPU codes only"
	@echo "  gpu     : Fast batch compile of GPU codes only"
	@echo "  serial        : Compile Serial"
	@echo "  serial_ultra  : Compile Serial Ultra"
	@echo "  openmp        : Compile OpenMP"
	@echo "  openmp_ultra  : Compile OpenMP Ultra"
	@echo "  mpi           : Compile MPI"
	@echo "  mpi_ultra     : Compile MPI Ultra"
	@echo "  hybrid        : Compile Hybrid (MPI+OpenMP)"
	@echo "  hybrid_ultra  : Compile Hybrid Ultra"
	@echo "  cuda          : Compile CUDA"
	@echo "  cuda_ultra    : Compile CUDA Ultra"
	@echo "  check_env     : Verify compiler versions"
	@echo "  clean         : Remove binaries"
