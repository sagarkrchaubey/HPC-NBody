#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <cstring> 
#include <cstdlib>

constexpr double DT   = 0.001;
constexpr double G    = 1.0;
constexpr double SOFT = 0.1;
constexpr int SEED    = 1234;
constexpr int BLOCK   = 256;

__global__ void nbody_kernel(
    double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ z,
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vz,
    double* __restrict__ m,
    int N)
{
    __shared__ double sx[BLOCK];
    __shared__ double sy[BLOCK];
    __shared__ double sz[BLOCK];
    __shared__ double sm[BLOCK];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double xi, yi, zi, vxi, vyi, vzi;
    
    if(i < N) {
        xi = x[i]; yi = y[i]; zi = z[i];
        vxi = vx[i]; vyi = vy[i]; vzi = vz[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for(int tile = 0; tile < N; tile += BLOCK)
    {
        int j = tile + threadIdx.x;
        
        if(j < N){
            sx[threadIdx.x] = x[j];
            sy[threadIdx.x] = y[j];
            sz[threadIdx.x] = z[j];
            sm[threadIdx.x] = m[j];
        } else {
            sx[threadIdx.x] = 0.0;
            sy[threadIdx.x] = 0.0;
            sz[threadIdx.x] = 0.0;
            sm[threadIdx.x] = 0.0;
        }
        
        __syncthreads();

        if(i < N) {
            #pragma unroll 8
            for(int k = 0; k < BLOCK; ++k)
            {
                double dx = sx[k] - xi;
                double dy = sy[k] - yi;
                double dz = sz[k] - zi;

                double dist2 = dx*dx + dy*dy + dz*dz + SOFT;
                
                double inv   = rsqrt(dist2); 
                double inv3  = inv * inv * inv;

                double f = G * sm[k] * inv3;

                Fx += f * dx;
                Fy += f * dy;
                Fz += f * dz;
            }
        }
        __syncthreads();
    }

    if(i < N) {
        vxi += Fx * DT;
        vyi += Fy * DT;
        vzi += Fz * DT;

        xi += vxi * DT;
        yi += vyi * DT;
        zi += vzi * DT;

        x[i] = xi;  y[i] = yi;  z[i] = zi;
        vx[i] = vxi; vy[i] = vyi; vz[i] = vzi;
    }
}

enum Mode { BENCH, VISUAL };

int main(int argc, char** argv)
{
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <N> <STEPS> <bench|visual> [SAVE_INTERVAL]\n";
        return 1;
    }

    const int N = atoi(argv[1]);
    const int STEPS = atoi(argv[2]);
    
    Mode mode = (strcmp(argv[3], "visual") == 0) ? VISUAL : BENCH;
    int saveInterval = (mode == VISUAL && argc > 4) ? atoi(argv[4]) : 1;

    size_t bytes = N * sizeof(double);

    std::vector<double> x(N), y(N), z(N), vx(N), vy(N), vz(N), m(N);

    srand(SEED);
    for(int i=0;i<N;i++){
        x[i]  = rand()/(double)RAND_MAX * 2.0 - 1.0;
        y[i]  = rand()/(double)RAND_MAX * 2.0 - 1.0;
        z[i]  = rand()/(double)RAND_MAX * 2.0 - 1.0;
        vx[i] = rand()/(double)RAND_MAX - 0.5;
        vy[i] = rand()/(double)RAND_MAX - 0.5;
        vz[i] = rand()/(double)RAND_MAX - 0.5;
        m[i]  = rand()/(double)RAND_MAX * 0.9 + 0.1;
    }

    double *dx,*dy,*dz,*dvx,*dvy,*dvz,*dm;
    cudaMalloc(&dx,bytes); cudaMalloc(&dy,bytes); cudaMalloc(&dz,bytes);
    cudaMalloc(&dvx,bytes); cudaMalloc(&dvy,bytes); cudaMalloc(&dvz,bytes);
    cudaMalloc(&dm,bytes);

    cudaMemcpy(dx,x.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,y.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dz,z.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dvx,vx.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dvy,vy.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dvz,vz.data(),bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(dm,m.data(),bytes,cudaMemcpyHostToDevice);

    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    std::ofstream out;
    if(mode == VISUAL){
        out.open("nbody_cuda_ultra_output.csv");
        out << "step,i,x,y,z,vx,vy,vz,m\n";
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int s=0;s<STEPS;s++){
        nbody_kernel<<<grid,block>>>(dx,dy,dz,dvx,dvy,dvz,dm,N);

        if(mode == VISUAL && s % saveInterval == 0){
            cudaMemcpy(x.data(), dx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(y.data(), dy, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(z.data(), dz, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(vx.data(), dvx, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(vy.data(), dvy, bytes, cudaMemcpyDeviceToHost);
            cudaMemcpy(vz.data(), dvz, bytes, cudaMemcpyDeviceToHost);

            for(int k=0;k<N;k++)
                out<<s<<","<<k<<","<<x[k]<<","<<y[k]<<","<<z[k]<<","<<vx[k]<<","<<vy[k]<<","<<vz[k]<<","<<m[k]<<"\n";
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms,start,stop);
    double sec = ms / 1000.0;
    double ops = (double)N * N * 20.0 * STEPS;

    std::cout << "Time:   " << sec << " s\n";
    std::cout << "GFLOPs: " << (ops / sec) / 1e9 << "\n";

    if(mode == VISUAL) out.close();

    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    cudaFree(dvx); cudaFree(dvy); cudaFree(dvz); cudaFree(dm);
    
    return 0;
}
