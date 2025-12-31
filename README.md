# 🌌 N-Body Simulation — High Performance Computing Project

> **Layered Performance Analysis and Optimization of the N-Body Problem Using Parallel Programming Paradigms**

A complete **HPC benchmark project** exploring how performance evolves when the same computational problem is implemented and optimized across multiple parallel computing models.

---

## 🚀 Project Overview

The **N-Body problem** models how particles interact under gravitational forces.  
This project focuses not on physics discovery, but on **performance engineering**:

> **How fast can we make the same algorithm run using modern parallel computing techniques?**

We start from a clean serial baseline and progressively apply advanced optimization and parallelization strategies across CPU and GPU architectures.

---

## 🧩 Implemented Variants

| Paradigm | Description |
|---------|-------------|
| 🧱 **Serial** | Baseline implementation |
| 🧵 **OpenMP** | Shared-memory parallelization |
| 🌐 **MPI** | Distributed-memory parallelization |
| 🔀 **Hybrid MPI + OpenMP** | Multi-level parallelism |
| 🎮 **CUDA** | GPU acceleration |

Each variant has:
- **Baseline version**
- **Highly optimized version**
- **Dedicated benchmarks & analysis**

---

## 🛠️ Optimization Techniques Used

- Data layout transformation (AoS → SoA)
- Cache blocking & memory alignment
- Loop unrolling & vectorization
- AVX-512 intrinsics (SIMD)
- NUMA-aware execution
- OpenMP scheduling & affinity tuning
- MPI topology optimization
- CUDA shared memory tiling
- Kernel fusion & memory coalescing
- Aggressive compiler optimizations & profiling-guided tuning

---

## 🧪 Benchmarking & Profiling

Benchmarks capture:
- Execution time
- Speedup & efficiency
- Strong & weak scaling behavior
- CPU & GPU utilization
- Memory bandwidth & cache behavior
- Communication overhead

Profiling tools:
- Intel VTune
- NVIDIA Nsight
- SLURM job instrumentation

---

## 🖥️ Hardware Environment

**Paramutkarsh HPC Cluster (C-DAC)**

| Component | Specification |
|----------|---------------|
| CPU | Intel Xeon Platinum 8268 (48 cores, dual-socket) |
| GPU | NVIDIA Tesla V100 (16GB) |
| Nodes | 156 total (CPU + GPU) |
| Interconnect | Mellanox InfiniBand 100 Gbps |
| OS | CentOS 7 |
| Scheduler | SLURM |

---

## 📁 Repository Structure
.
├── src/ # Source codes for all variants
├── bin/ # Compiled executables
├── scripts/ # SLURM job scripts & automation
├── logs/ # Benchmark & profiler outputs
├── data/ # Simulation data & results
└── report/ # Project documentation

---

## 🧠 What This Project Demonstrates

- Practical application of HPC concepts  
- How architectural awareness improves performance  
- Trade-offs between parallel programming models  
- Real-world performance engineering workflow  

---

## 🧬 How to Build & Run

### Compile (example)
```bash
make all
make cpu
make gpu
make serial
make openmp
make mpi
make hybrid
make cuda

Run on cluster
sbatch scripts/run_openmp.sh
sbatch scripts/run_mpi.sh
sbatch scripts/run_cuda.sh


📊 Final Goal
To build a comprehensive performance study showing:
How the same N-Body algorithm evolves from baseline code into a finely-tuned high-performance machine.
