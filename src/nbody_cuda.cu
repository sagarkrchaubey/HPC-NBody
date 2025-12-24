#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <cuda_runtime.h>

const double DT = 0.001;
const double G = 1.0;
const double SOFTENING = 0.1;
const int SEED = 1234;
const int THREADS_PER_BLOCK = 256;

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void initialize_bodies(std::vector<Body>& bodies) {
    std::srand(SEED); 
    int N = bodies.size();
    for (int i = 0; i < N; i++) {
        bodies[i].x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        bodies[i].y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        bodies[i].z = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        bodies[i].vx = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].vy = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].vz = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].mass = ((double)rand() / RAND_MAX) * 0.9 + 0.1;
    }
}

__global__ void update_physics_kernel(Body* bodies, int N, double dt, double g, double softening) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
        double body_ix = bodies[i].x;
        double body_iy = bodies[i].y;
        double body_iz = bodies[i].z;
        double body_imass = bodies[i].mass;

        for (int j = 0; j < N; j++) {
            if (i == j) continue;

            double dx = bodies[j].x - body_ix;
            double dy = bodies[j].y - body_iy;
            double dz = bodies[j].z - body_iz;
            
            double distSq = dx*dx + dy*dy + dz*dz + softening;
            double dist = sqrt(distSq);
            double f = (g * body_imass * bodies[j].mass) / (dist * dist * dist);

            fx += f * dx;
            fy += f * dy;
            fz += f * dz;
        }

        double ax = fx / body_imass;
        double ay = fy / body_imass;
        double az = fz / body_imass;

        bodies[i].vx += ax * dt;
        bodies[i].vy += ay * dt;
        bodies[i].vz += az * dt;

        bodies[i].x += bodies[i].vx * dt;
        bodies[i].y += bodies[i].vy * dt;
        bodies[i].z += bodies[i].vz * dt;
    }
}

int main(int argc, char* argv[]) {
    int n_bodies = 100;
    int n_steps = 1000;
    bool benchmark_mode = false;

    if (argc >= 2) n_bodies = std::atoi(argv[1]);
    if (argc >= 3) n_steps = std::atoi(argv[2]);
    if (argc >= 4 && std::string(argv[3]) == "true") benchmark_mode = true;

    std::vector<Body> h_bodies(n_bodies);
    initialize_bodies(h_bodies);

    Body* d_bodies;
    size_t size = n_bodies * sizeof(Body);
    gpuErrchk(cudaMalloc((void**)&d_bodies, size));
    
    gpuErrchk(cudaMemcpy(d_bodies, h_bodies.data(), size, cudaMemcpyHostToDevice));

    int blocksPerGrid = (n_bodies + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    std::ofstream outfile;
    std::string filename = "nbody_output_cuda_N" + std::to_string(n_bodies) 
                         + "_S" + std::to_string(n_steps) 
                         + "_TPB" + std::to_string(THREADS_PER_BLOCK) + ".csv";

    if (!benchmark_mode) {
        outfile.open(filename.c_str());
        outfile << "Step,BodyID,X,Y,Z,Mass" << std::endl;
        std::cout << "--- CUDA Simulation Initialized ---" << std::endl;
        std::cout << "Bodies: " << n_bodies << " | Block Size: " << THREADS_PER_BLOCK << std::endl;
        std::cout << "Output File: " << filename << std::endl;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);

    for (int step = 0; step < n_steps; step++) {
        
        if (!benchmark_mode && step % 1 == 0) {
            gpuErrchk(cudaMemcpy(h_bodies.data(), d_bodies, size, cudaMemcpyDeviceToHost));
            for (int i = 0; i < n_bodies; i++) {
                outfile << step << "," << i << "," 
                        << h_bodies[i].x << "," << h_bodies[i].y << "," << h_bodies[i].z 
                        << "," << h_bodies[i].mass << "\n";
            }
        }

        update_physics_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_bodies, n_bodies, DT, G, SOFTENING);
        gpuErrchk(cudaPeekAtLastError());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double seconds = milliseconds / 1000.0;

    if (!benchmark_mode) outfile.close();

    double total_ops = (double)n_bodies * n_bodies * 20.0 * n_steps;
    double gflops = (total_ops / seconds) / 1e9;

    std::cout << "\n--- CUDA Simulation Results ---" << std::endl;
    std::cout << "Time:  " << seconds << " s" << std::endl;
    std::cout << "Perf:  " << gflops << " GFLOPs" << std::endl;

    cudaFree(d_bodies);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
