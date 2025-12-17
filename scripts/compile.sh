#!/bin/bash

TARGET=$1

if [ -z "$TARGET" ]; then
    echo "Usage: $0 [all|serial|openmp|mpi|hybrid|cuda]"
    exit 1
fi

BASE_DIR=".."
BIN_DIR="$BASE_DIR/bin"
SRC_DIR="$BASE_DIR/src"
FLAGS="-std=c++11 -g -fno-omit-frame-pointer"

mkdir -p $BIN_DIR

module purge
module load openmpi/4.1.1
module load cuda/12.0

export OMPI_MCA_btl=tcp,self

case $TARGET in
    serial)
        g++ $FLAGS $SRC_DIR/serial/nbody_serial.cpp -o $BIN_DIR/nbody_serial
        ;;
    openmp)
        g++ $FLAGS -fopenmp $SRC_DIR/openmp/nbody_openmp.cpp -o $BIN_DIR/nbody_openmp
        ;;
    mpi)
        mpicxx $FLAGS $SRC_DIR/mpi/nbody_mpi.cpp -o $BIN_DIR/nbody_mpi
        ;;
    hybrid)
        mpicxx $FLAGS -fopenmp $SRC_DIR/hybrid/nbody_hybrid.cpp -o $BIN_DIR/nbody_hybrid
        ;;
    cuda)
        nvcc -std=c++11 -g -lineinfo $SRC_DIR/cuda/nbody_cuda.cu -o $BIN_DIR/nbody_cuda
        ;;
    all)
        g++ $FLAGS $SRC_DIR/serial/nbody_serial.cpp -o $BIN_DIR/nbody_serial
        g++ $FLAGS -fopenmp $SRC_DIR/openmp/nbody_openmp.cpp -o $BIN_DIR/nbody_openmp
        mpicxx $FLAGS $SRC_DIR/mpi/nbody_mpi.cpp -o $BIN_DIR/nbody_mpi
        mpicxx $FLAGS -fopenmp $SRC_DIR/hybrid/nbody_hybrid.cpp -o $BIN_DIR/nbody_hybrid
        nvcc -std=c++11 -g -lineinfo $SRC_DIR/cuda/nbody_cuda.cu -o $BIN_DIR/nbody_cuda
        ;;
    *)
        echo "Invalid option: $TARGET"
        echo "Available options: all, serial, openmp, mpi, hybrid, cuda"
        exit 1
        ;;
esac
