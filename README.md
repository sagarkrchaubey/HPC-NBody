# 🌌 HPC N-Body Simulation

### Layered Performance Analysis and Optimization of the N-Body Problem Using Parallel Programming Paradigms

> **The core question this project answers:**
> *How fast can the exact same gravitational simulation run when you apply increasingly powerful parallel hardware and optimization strategies — and what does each architecture actually contribute?*

<br>

[![Language](https://img.shields.io/badge/Language-C%2B%2B17%20%7C%20CUDA-blue?style=flat-square)](https://isocpp.org/)
[![Parallelism](https://img.shields.io/badge/Parallelism-OpenMP%20%7C%20MPI%20%7C%20CUDA-green?style=flat-square)](https://www.openmp.org/)
[![Platform](https://img.shields.io/badge/Platform-Param%20Utkarsh%20%7C%20C--DAC-orange?style=flat-square)](https://www.cdac.in/)
[![Scheduler](https://img.shields.io/badge/Scheduler-SLURM-red?style=flat-square)](https://slurm.schedmd.com/)
[![Profiling](https://img.shields.io/badge/Profiling-Intel%20VTune%20%7C%20NVIDIA%20Nsight-purple?style=flat-square)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)

<br>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [The N-Body Problem](#-the-n-body-problem)
3. [Project Architecture](#-project-architecture)
4. [Implementation Details](#-implementation-details)
5. [Optimization Techniques](#-optimization-techniques)
6. [Hardware Environment](#-hardware-environment)
7. [Repository Structure](#-repository-structure)
8. [Prerequisites](#-prerequisites)
9. [Build Instructions](#-build-instructions)
10. [Running on the Cluster](#-running-on-the-cluster)
11. [Profiling](#-profiling)
12. [Visualization](#-visualization)
13. [Performance Results](#-performance-results)
14. [Key Findings](#-key-findings)
15. [Acknowledgments](#-acknowledgments)

---

## 🚀 Project Overview

This is a complete **HPC benchmarking project** built at C-DAC's Advanced Computing Training School and run on the **Param Utkarsh supercomputer**. The project implements the gravitational N-Body simulation five different ways — Serial, OpenMP, MPI, Hybrid (MPI+OpenMP), and CUDA — with each implementation written as a **completely independent program**.

Each of the 5 paradigms has **two versions**:
- A **naive baseline** — straightforward, unoptimized implementation
- An **ultra-optimized version** — hand-tuned with architecture-specific optimizations

This gives **10 standalone source files** in total, benchmarked separately on the cluster to isolate exactly what each paradigm and each optimization technique contributes.

The project is **not a heterogeneous system**. It is a controlled performance study — each binary runs independently on its appropriate hardware (CPU or GPU partition), and results are compared after the fact.

---

## 🪐 The N-Body Problem

The N-Body problem simulates how N particles interact under gravitational force. Every particle exerts a gravitational pull on every other particle, making it an **O(N²) problem** — the number of calculations grows as the square of particle count.

For N particles over S time steps, the total floating-point operations are:

```
Total FLOPS = N × N × 20 × S
```

The 20 comes from the operations per particle pair: 3 subtractions (dx, dy, dz), 3 multiplications + 2 additions (distSq), 1 sqrt, 2 multiplications (force), 6 multiply-adds (force components), and the velocity/position updates.

This makes N-Body an excellent HPC benchmark — it is compute-bound, memory-bound at scale, and has clear bottlenecks that respond differently to CPU parallelism vs GPU parallelism.

**Physics constants used across all implementations:**
```
G           = 1.0       (gravitational constant)
DT          = 0.001     (timestep)
SOFTENING   = 0.1       (prevents singularity at r → 0)
SEED        = 1234      (deterministic initialization)
```

---

## 🏗️ Project Architecture

The project is structured as a layered benchmark — each layer adds a new level of parallelism or optimization, and performance is measured at every step.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAME ALGORITHM, 5 PARADIGMS                  │
│                    Each = Separate Independent Binary           │
├──────────┬──────────┬──────────┬──────────┬────────────────────┤
│  Serial  │  OpenMP  │   MPI    │  Hybrid  │       CUDA         │
│          │ Shared   │Distributed│MPI+OpenMP│    GPU (V100)      │
│ 1 Core   │ Memory   │  Memory  │Multi-lvl │  2560 CUDA Cores   │
├──────────┴──────────┴──────────┴──────────┴────────────────────┤
│         Each paradigm has: Naive + Ultra (10 files total)       │
└─────────────────────────────────────────────────────────────────┘
```

### The 10 Source Files

| File | Paradigm | Version | Key Technique |
|------|----------|---------|---------------|
| `nbody_serial.cpp` | Serial | Naive | AoS layout, scalar math |
| `nbody_serial_ultra.cpp` | Serial | Optimized | SoA layout, AVX-512 SIMD |
| `nbody_openmp.cpp` | OpenMP | Naive | `#pragma omp parallel for` |
| `nbody_openmp_ultra.cpp` | OpenMP | Optimized | NUMA first-touch, static scheduling |
| `nbody_mpi.cpp` | MPI | Naive | `MPI_Allgather`, AoS |
| `nbody_mpi_ultra.cpp` | MPI | Optimized | Ring topology, blocking sends |
| `nbody_hybrid.cpp` | Hybrid | Naive | MPI + OpenMP, `MPI_THREAD_FUNNELED` |
| `nbody_hybrid_ultra.cpp` | Hybrid | Optimized | AVX-512 + ring + NUMA first-touch |
| `nbody_cuda.cu` | CUDA | Naive | AoS, global memory, 256 threads/block |
| `nbody_cuda_ultra.cu` | CUDA | Optimized | SoA, shared memory tiling, `rsqrt()` |

---

## 🔬 Implementation Details

### 1. Serial

The baseline. A single thread on a single core computes all N×N force interactions.

- **Naive:** Bodies stored as Array of Structs (AoS). Each iteration reads interleaved `x,y,z,vx,vy,vz,mass` from memory, causing poor cache line utilization.
- **Ultra:** Bodies reorganized into Struct of Arrays (SoA) — separate contiguous arrays for `x[]`, `y[]`, `z[]`, etc. This enables the compiler to auto-vectorize and allows explicit AVX-512 SIMD intrinsics.

### 2. OpenMP (Shared Memory)

All 48 cores of a single dual-socket Intel Xeon node work on the same problem, sharing memory.

- **Naive:** `#pragma omp parallel for` over the outer loop. Simple and effective.
- **Ultra:** NUMA-aware first-touch initialization — threads write to their own memory pages at startup, forcing the OS to allocate pages on the correct NUMA node (socket). Thread affinity pinned via `OMP_PROC_BIND=close` and `OMP_PLACES=cores`. Static scheduling for load balance. 48 threads total.

### 3. MPI (Distributed Memory)

Each MPI rank owns a slice of the particles. Force calculation is done locally, then all ranks synchronize position data.

- **Naive:** `MPI_Allgather` to broadcast positions to all ranks after each step. 48 MPI ranks.
- **Ultra:** Ring topology communication — each rank exchanges data only with its neighbors in a ring, reducing global synchronization overhead. Blocking `MPI_Sendrecv` for overlap of communication and computation.

### 4. Hybrid (MPI + OpenMP)

Two-level parallelism: MPI distributes work across sockets, OpenMP threads parallelize within each socket.

- **Configuration:** 2 MPI ranks × 24 OpenMP threads/rank = 48 total cores.
- **Socket mapping:** Each MPI rank is pinned to one physical socket via `--map-by socket:PE=24`, ensuring threads stay local to their NUMA domain.
- **Thread safety:** Uses `MPI_THREAD_FUNNELED` — only the master thread makes MPI calls, which is the safest and most performant model for this workload.
- **Ultra:** Adds AVX-512 intrinsics (`_mm512_*`) for the inner force loop, ring-topology MPI communication, 64-byte aligned memory allocation (`aligned_alloc(64, ...)`), and NUMA first-touch initialization via the OpenMP parallel region.

### 5. CUDA (GPU)

Massively parallel execution on the NVIDIA Tesla V100.

- **Naive:** Each CUDA thread handles one particle (body `i`). Bodies stored as AoS in global memory. 256 threads per block. Standard `sqrt()`.
- **Ultra:** Full SoA data layout. **Shared memory tiling** — each block of 256 threads cooperatively loads a tile of particle data (`sx[]`, `sy[]`, `sz[]`, `sm[]`) into `__shared__` memory before computing forces, reusing L1/shared memory instead of hitting global memory N times. Uses `rsqrt()` (hardware fast reciprocal square root). `#pragma unroll 8` on the inner tile loop. `__restrict__` pointers to inform the compiler there is no aliasing.

---

## ⚙️ Optimization Techniques

| Technique | Where Applied | What It Does |
|-----------|--------------|--------------|
| **AoS → SoA** | Serial Ultra, CUDA Ultra | Improves memory access locality; enables SIMD vectorization |
| **AVX-512 Intrinsics** | Serial Ultra, Hybrid Ultra | Processes 8 doubles in a single instruction using 512-bit SIMD |
| **`-march=cascadelake`** | All CPU builds | Emits instructions tuned for Cascade Lake microarchitecture |
| **`-mprefer-vector-width=512`** | All CPU Ultra builds | Forces auto-vectorizer to prefer 512-bit vectors |
| **`-ffast-math` + loop unrolling** | All Ultra builds | Relaxes IEEE strictness for speed; unrolls loops to reduce branch overhead |
| **NUMA First-Touch** | OpenMP Ultra, Hybrid Ultra | Forces memory pages to allocate on the local NUMA socket |
| **Thread Affinity** | OpenMP, Hybrid | `OMP_PROC_BIND=close` + `OMP_PLACES=cores` prevents thread migration |
| **MPI Ring Topology** | MPI Ultra, Hybrid Ultra | Replaces global `Allgather` with nearest-neighbor ring sends |
| **`MPI_THREAD_FUNNELED`** | Hybrid | Only master thread calls MPI; avoids locking overhead |
| **64-byte Aligned Memory** | Hybrid Ultra | `aligned_alloc(64,...)` ensures AVX-512 loads are aligned |
| **Shared Memory Tiling** | CUDA Ultra | Each block loads a 256-body tile into shared memory, reducing global memory bandwidth |
| **`rsqrt()` + `__restrict__`** | CUDA Ultra | Hardware fast inverse sqrt; no-alias hint for compiler |
| **Kernel Fusion** | CUDA | Force + velocity + position update done in a single kernel launch |
| **`--use_fast_math`** | CUDA Ultra | Compiler flag for approximate math operations on GPU |

---

## 🖥️ Hardware Environment

All benchmarks ran on **C-DAC's Param Utkarsh HPC Cluster**.

| Component | Specification |
|-----------|--------------|
| **CPU** | Intel Xeon Platinum 8268 (Cascade Lake) |
| **CPU Cores** | 48 cores per node (dual-socket, 24 cores/socket) |
| **GPU** | NVIDIA Tesla V100 (16 GB HBM2) |
| **GPU CUDA Cores** | 2,560 |
| **Interconnect** | Mellanox InfiniBand 100 Gbps |
| **Total Nodes** | 156 (CPU + GPU partitions) |
| **Operating System** | CentOS 7 |
| **Job Scheduler** | SLURM |

### Software Stack

| Component | Version |
|-----------|---------|
| GCC (CPU builds) | 13.1.0 |
| GCC (GPU builds) | 8.5.0 |
| OpenMPI | 4.1.1 |
| CUDA Toolkit | 11.0 / 12.0 |
| Intel VTune | 2021.7.1 |
| NVIDIA Nsight Systems | Latest available |
| Spack | For environment management |

---

## 📁 Repository Structure

```
HPC-NBody/
│
├── src/                          # All 10 source files
│   ├── nbody_serial.cpp          # Serial — naive baseline
│   ├── nbody_serial_ultra.cpp    # Serial — AVX-512 + SoA optimized
│   ├── nbody_openmp.cpp          # OpenMP — naive (48 threads)
│   ├── nbody_openmp_ultra.cpp    # OpenMP — NUMA-aware + static scheduling
│   ├── nbody_mpi.cpp             # MPI — naive (48 ranks, Allgather)
│   ├── nbody_mpi_ultra.cpp       # MPI — ring topology optimized
│   ├── nbody_hybrid.cpp          # Hybrid — naive (2 ranks × 24 threads)
│   ├── nbody_hybrid_ultra.cpp    # Hybrid — AVX-512 + ring + NUMA
│   ├── nbody_cuda.cu             # CUDA — naive (global memory, AoS)
│   └── nbody_cuda_ultra.cu       # CUDA — shared memory tiling + SoA
│
├── scripts/                      # 20 SLURM scripts (run + profile)
│   ├── run_serial.sh             # SLURM: run serial baseline
│   ├── run_serial_ultra.sh       # SLURM: run serial ultra
│   ├── run_openmp.sh             # SLURM: run OpenMP (48 threads)
│   ├── run_openmp_ultra.sh       # SLURM: run OpenMP ultra
│   ├── run_mpi.sh                # SLURM: run MPI (48 ranks)
│   ├── run_mpi_ultra.sh          # SLURM: run MPI ultra
│   ├── run_hybrid.sh             # SLURM: run Hybrid (2×24)
│   ├── run_hybrid_ultra.sh       # SLURM: run Hybrid ultra
│   ├── run_cuda.sh               # SLURM: run CUDA on GPU node
│   ├── run_cuda_ultra.sh         # SLURM: run CUDA ultra on GPU node
│   ├── prof_serial.sh            # VTune hotspot profile: serial
│   ├── prof_serial_ultra.sh      # VTune hotspot profile: serial ultra
│   ├── prof_openmp.sh            # VTune hotspot profile: OpenMP
│   ├── prof_openmp_ultra.sh      # VTune hotspot profile: OpenMP ultra
│   ├── prof_mpi.sh               # VTune hotspot profile: MPI
│   ├── prof_mpi_ultra.sh         # VTune hotspot profile: MPI ultra
│   ├── prof_hybrid.sh            # VTune hotspot profile: Hybrid
│   ├── prof_hybrid_ultra.sh      # VTune hotspot profile: Hybrid ultra
│   ├── prof_cuda.sh              # Nsight Systems profile: CUDA
│   ├── prof_cuda_ultra.sh        # Nsight Systems profile: CUDA ultra
│   └── spec.sh                   # Hardware spec dump (CPU + GPU nodes)
│
├── viz/                          # Visualization scripts
│   └── render_galaxy.py          # Python: renders simulation to MP4
│                                 # (3D rotating view + density heatmap
│                                 #  + velocity distribution)
│
├── bin/                          # Compiled binaries (git-ignored)
├── logs/                         # SLURM job logs and error files
├── vtune_reports/                # Intel VTune profiling results
├── cuda_reports/                 # NVIDIA Nsight .nsys-rep files
│
├── Makefile                      # Build system for all 10 variants
├── Nbody report.pdf              # Full project report
└── README.md                     # This file
```

---

## 📦 Prerequisites

### For CPU Builds
- GCC 13.x with C++17 support
- OpenMPI 4.x (`mpicxx` wrapper)
- Intel oneAPI VTune (for profiling)
- Spack (for environment management on the cluster)

### For GPU Builds
- GCC 8.5 (CUDA 11.0 compatible host compiler)
- CUDA Toolkit 11.0+ (`nvcc`)
- NVIDIA Nsight Systems (`nsys`)

### For Visualization
```bash
pip install numpy matplotlib scipy pandas ffmpeg-python
# FFmpeg must be installed and available in PATH
```

---

## 🔨 Build Instructions

The Makefile handles all 10 variants. It sets up separate compiler environments for CPU and GPU targets automatically.

### Build Everything
```bash
make all        # Compiles all 10 binaries (cpu + gpu)
```

### Build CPU or GPU Only
```bash
make cpu        # Compiles all 8 CPU variants in one shot
make gpu        # Compiles both CUDA variants
```

### Build a Specific Variant
```bash
make serial           # Serial naive
make serial_ultra     # Serial with AVX-512 + SoA
make openmp           # OpenMP naive
make openmp_ultra     # OpenMP NUMA-aware
make mpi              # MPI naive (48 ranks)
make mpi_ultra        # MPI with ring topology
make hybrid           # Hybrid naive (2 ranks × 24 threads)
make hybrid_ultra     # Hybrid with AVX-512 + ring + NUMA
make cuda             # CUDA naive
make cuda_ultra       # CUDA shared memory tiling
```

### Utility Targets
```bash
make check_env  # Verify compiler versions for both environments
make clean      # Remove all compiled binaries
make dir        # Create required output directories
make help       # Show all available targets
```

---

## 🚀 Running on the Cluster

All executables accept the same command-line interface:
```
./bin/<binary> <N_PARTICLES> <N_STEPS> <bench|visual> [SAVE_INTERVAL]
```

| Argument | Description |
|----------|-------------|
| `N_PARTICLES` | Number of bodies in the simulation |
| `N_STEPS` | Number of time steps to simulate |
| `bench` | Benchmark mode — prints Time and GFLOPs only |
| `visual` | Visual mode — writes particle data to CSV each step |
| `SAVE_INTERVAL` | (Visual mode only) Save every N-th step |

### Submitting SLURM Jobs

Each variant has a dedicated run script. Submit it with `sbatch`:

```bash
# Serial
sbatch scripts/run_serial.sh 5000 1000 bench

# OpenMP (48 threads, 1 node)
sbatch scripts/run_openmp.sh 5000 1000 bench

# MPI (48 ranks, 1 node)
sbatch scripts/run_mpi.sh 5000 1000 bench

# Hybrid (2 MPI ranks × 24 OpenMP threads)
sbatch scripts/run_hybrid.sh 5000 1000 bench

# CUDA (GPU partition, 1 V100)
sbatch scripts/run_cuda.sh 20000 1000 bench
```

You can also pass arguments directly:
```bash
sbatch scripts/run_cuda_ultra.sh 30000 1000 bench
#                                ^N    ^steps ^mode
```

### Generating Visualization Data

Run any implementation in `visual` mode to generate a CSV:
```bash
sbatch scripts/run_openmp.sh 2000 1000 visual 5
# Saves particle positions every 5 steps to: nbody_output_openmp_N2000.csv
```

---

## 🔍 Profiling

Each implementation has a matching profiling script. These are separate from the run scripts and wrap the binary with the appropriate profiler.

### CPU Profiling (Intel VTune — Hotspot Analysis)

```bash
# Profile any CPU variant
sbatch scripts/prof_serial.sh 5000
sbatch scripts/prof_openmp.sh 5000
sbatch scripts/prof_mpi.sh 5000
sbatch scripts/prof_hybrid.sh 5000

# Profile the optimized versions
sbatch scripts/prof_serial_ultra.sh 5000
sbatch scripts/prof_openmp_ultra.sh 5000
sbatch scripts/prof_mpi_ultra.sh 5000
sbatch scripts/prof_hybrid_ultra.sh 5000
```

VTune results are saved to `vtune_reports/<variant>_N<n>_ID<jobid>/`. These can be opened with the VTune GUI or analyzed via CLI:
```bash
vtune -report hotspots -result-dir vtune_reports/<result_dir>
```

VTune is also invoked inside `mpirun` for MPI variants to trace all ranks:
```bash
mpirun -np 48 vtune -collect hotspots -result-dir <dir> -- ./bin/nbody_mpi ...
```

### GPU Profiling (NVIDIA Nsight Systems)

```bash
sbatch scripts/prof_cuda.sh 20000
sbatch scripts/prof_cuda_ultra.sh 20000
```

Nsight reports are saved to `cuda_reports/<name>.nsys-rep`. Open with:
```bash
nsys-ui cuda_reports/<report>.nsys-rep
```

The profiling scripts trace CUDA API calls, OS runtime, and NVTX markers:
```bash
nsys profile --stats=true --trace=cuda,osrt,nvtx --output=<report> ./bin/nbody_cuda ...
```

### Hardware Topology Inspection

```bash
bash scripts/spec.sh
# Dumps: lscpu, NUMA topology, memory info, InfiniBand status, GPU details
# for both CPU and GPU compute nodes
```

---

## 📊 Visualization

The `viz/render_galaxy.py` script takes CSV output from any simulation run and renders it into a multi-panel video.

### Generate a CSV First
```bash
sbatch scripts/run_openmp.sh 2000 1000 visual 5
# Produces: nbody_output_openmp_N2000.csv
```

### Run the Renderer
```bash
cd viz/
python render_galaxy.py
```

Configure the script at the top:
```python
CSV_FILE   = "nbody_output_openmp_N2000.csv"
OUT_DIR    = "frames_final_v2"
VIDEO_NAME = "galaxy_simulation.mp4"
STEP_STRIDE = 1   # Process every N-th timestep
FPS         = 30
```

### Video Output

The renderer produces a 4-panel visualization:

| Panel | Description |
|-------|-------------|
| **Global View** (left) | Full 3D rotating view of all particles, colored by local density |
| **Tracking Core** (center) | Zoomed-in view of the gravitational core, with magnification indicator |
| **XY Projection Density** (top right) | 2D hex-bin density heatmap showing structure formation |
| **Velocity Distribution** (bottom right) | Histogram of particle speed magnitudes across the timestep |

Frames are rendered in parallel using Python `multiprocessing.Pool` and encoded to MP4 via FFmpeg.

---

## 📈 Performance Results

All benchmarks run on Param Utkarsh with N = 5,000–30,000 particles, 1,000 steps. GFLOPS calculated as:
```
GFLOPS = (N × N × 20 × STEPS) / (time_seconds × 1e9)
```

### Peak Performance — Optimized Variants

| Paradigm | Key Technique | Peak GFLOPS |
|----------|--------------|:-----------:|
| 🧱 Naive Serial | Scalar, Array of Structs | 5.1 |
| ⚡ Optimized Serial | AVX-512, Struct of Arrays | 45.4 |
| 🌐 MPI Distributed | Ring Topology, Blocking | 715.7 |
| 🧵 OpenMP Shared | NUMA First-Touch, Static | 1,378.2 |
| 🔀 Hybrid MPI+OpenMP | Thread Funneled + Ring | 1,425.9 |
| 🎮 CUDA GPU | Shared Memory Tiling | **3,853.1** |

### Performance Scaling — Speedup vs. Naive Serial Baseline

```
Naive Serial    ██ 5.1 GFLOPS         (1×)
Opt. Serial     █████████ 45.4        (8.9×)
MPI             ████████████████████████████████████ 715.7   (140×)
OpenMP          ███████████████████████████████████████████████████████████████ 1,378.2  (270×)
Hybrid          ██████████████████████████████████████████████████████████████████ 1,425.9 (279×)
CUDA            ████████████████████████████████████████████████████████████████████████████████████████████████████████ 3,853.1 (755×)
```

### Key Speedup Numbers

| Comparison | Speedup |
|------------|---------|
| Serial → Optimized Serial | **8.9×** (AVX-512 only, no extra cores) |
| Serial → MPI (48 ranks) | **140×** |
| Serial → OpenMP (48 threads) | **270×** |
| OpenMP → Hybrid | **1.03×** (small overhead from MPI layer) |
| Serial → CUDA | **755×** |
| MPI → CUDA | **5.4×** |

---

## 💡 Key Findings

**1. SIMD optimization alone gives nearly 9× on a single core.**
The jump from 5.1 to 45.4 GFLOPS comes purely from AVX-512 vectorization and the AoS→SoA layout change — no additional hardware used.

**2. OpenMP outperforms MPI at the same core count.**
With 48 cores on a single node, OpenMP (1,378 GFLOPS) beats MPI (715.7 GFLOPS). Shared memory removes all communication overhead that MPI's `Allgather` incurs.

**3. Hybrid barely improves over OpenMP for this problem.**
Hybrid (1,425.9 GFLOPS) is only marginally better than OpenMP (1,378.2 GFLOPS) on a single node. Hybrid parallelism pays off significantly when scaling across multiple nodes — on a single node, the MPI overhead nearly cancels out the gains from socket-level locality.

**4. CUDA dominates everything at high N.**
At N = 20,000+, the V100's 2,560 CUDA cores and HBM2 memory bandwidth make shared memory tiling extremely effective. The jump from naive CUDA to CUDA Ultra (shared memory tiling) alone contributes a significant fraction of the GPU speedup.

**5. Optimization techniques compound.**
The Hybrid Ultra's combination of AVX-512 + ring topology + NUMA first-touch + aligned memory all contributing together is more than the sum of their individual parts.

---

## 🙏 Acknowledgments

- **C-DAC (Centre for Development of Advanced Computing)** — for providing access to the Param Utkarsh supercomputer under the ACC-HPC program
- **Intel VTune** and **NVIDIA Nsight Systems** — profiling tools that made the optimization loop possible
- The **N-Body simulation community** for well-established benchmarking methodology

---

<div align="center">

**Built on Param Utkarsh | C-DAC Advanced Computing Training School**

*"How fast can the same algorithm run? — Turns out, 755 times faster."*

</div>
