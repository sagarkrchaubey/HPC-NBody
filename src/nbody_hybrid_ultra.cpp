#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <algorithm>
#include <vector>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h> // For core checking

using namespace std;

constexpr double DT = 0.001;
constexpr double G  = 1.0;
constexpr double SOFT = 0.1;
constexpr int SEED = 1234;

enum Mode { BENCH, VISUAL };

// Helper to check core affinity
int get_core_id() { return sched_getcpu(); }

int main(int argc, char** argv)
{
    // Initialize MPI with Thread Support
    // MPI_THREAD_FUNNELED: Only the main thread makes MPI calls (safest/fastest)
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(argc < 4) {
        if(rank == 0) cerr << "Usage: " << argv[0] << " <N> <STEPS> <bench|visual> [SAVE_INTERVAL]\n";
        MPI_Finalize();
        return 1;
    }

    const int N_GLOBAL = atoi(argv[1]);
    const int STEPS = atoi(argv[2]);
    Mode mode = (strcmp(argv[3], "visual") == 0) ? VISUAL : BENCH;
    int saveInterval = (mode == VISUAL && argc > 4) ? atoi(argv[4]) : 1;

    // --- 1. Load Balancing ---
    int base_N = N_GLOBAL / size;
    int remainder = N_GLOBAL % size;
    int my_N = base_N + (rank < remainder ? 1 : 0);
    
    // Calculate max_N for buffer allocation
    int max_N = base_N + (remainder > 0 ? 1 : 0);

    // Global offset calculation (for RNG consistency)
    int my_offset = 0;
    std::vector<int> all_counts(size);
    std::vector<int> all_displs(size);
    int current_disp = 0;
    for(int r=0; r<size; r++) {
        all_counts[r] = base_N + (r < remainder ? 1 : 0);
        all_displs[r] = current_disp;
        if(r == rank) my_offset = current_disp;
        current_disp += all_counts[r];
    }

    // --- 2. Memory Allocation (Aligned for AVX-512) ---
    size_t bytes = my_N * sizeof(double);
    if (bytes % 64 != 0) bytes += 64 - (bytes % 64);
    
    size_t buf_bytes = max_N * sizeof(double);
    if (buf_bytes % 64 != 0) buf_bytes += 64 - (buf_bytes % 64);

    double* __restrict__ x  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ y  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ z  = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vx = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vy = (double*) aligned_alloc(64, bytes);
    double* __restrict__ vz = (double*) aligned_alloc(64, bytes);
    double* __restrict__ m  = (double*) aligned_alloc(64, bytes);

    // Ring Buffers
    double* send_x = (double*) aligned_alloc(64, buf_bytes);
    double* send_y = (double*) aligned_alloc(64, buf_bytes);
    double* send_z = (double*) aligned_alloc(64, buf_bytes);
    double* send_m = (double*) aligned_alloc(64, buf_bytes);

    double* recv_x = (double*) aligned_alloc(64, buf_bytes);
    double* recv_y = (double*) aligned_alloc(64, buf_bytes);
    double* recv_z = (double*) aligned_alloc(64, buf_bytes);
    double* recv_m = (double*) aligned_alloc(64, buf_bytes);

    // --- 3. NUMA First Touch Initialization ---
    // We spawn threads here to touch the memory, forcing allocation 
    // on the local socket (the socket where this MPI rank is pinned).
    #pragma omp parallel
    {
        // Debug: Check affinity (only on Rank 0)
        #pragma omp master
        {
            if(rank == 0) {
                // Verify we have 24 threads per rank (if properly configured)
                // cout << "Rank 0 Threads: " << omp_get_num_threads() << endl;
            }
        }
        
        // Static schedule distributes loop iterations to threads.
        // The thread that writes to index 'i' first causes the OS to allocate that page.
        #pragma omp for schedule(static)
        for(int i=0; i<my_N; i++) {
            x[i] = 0.0; y[i] = 0.0; z[i] = 0.0;
            vx[i] = 0.0; vy[i] = 0.0; vz[i] = 0.0; m[i] = 0.0;
        }
        
        // Also touch buffers (not strictly necessary but good practice)
        // We assume max_N is reasonable.
        #pragma omp for schedule(static)
        for(int i=0; i<max_N; i++) {
            send_x[i]=0; send_y[i]=0; send_z[i]=0; send_m[i]=0;
            recv_x[i]=0; recv_y[i]=0; recv_z[i]=0; recv_m[i]=0;
        }
    }

    // --- 4. RNG Initialization (Deterministic Serial) ---
    srand(SEED);
    for(int i=0; i<my_offset; i++) {
        rand(); rand(); rand(); rand(); rand(); rand(); rand();
    }
    for(int i=0; i<my_N; i++){
        x[i]=rand()/(double)RAND_MAX*2.0-1.0;
        y[i]=rand()/(double)RAND_MAX*2.0-1.0;
        z[i]=rand()/(double)RAND_MAX*2.0-1.0;
        vx[i]=rand()/(double)RAND_MAX-0.5;
        vy[i]=rand()/(double)RAND_MAX-0.5;
        vz[i]=rand()/(double)RAND_MAX-0.5;
        m[i]=rand()/(double)RAND_MAX*0.9+0.1;
    }

    ofstream out;
    if(mode == VISUAL && rank == 0) {
        out.open("nbody_output.csv");
        out << "step,i,x,y,z,vx,vy,vz,m\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    // --- 5. Main Loop ---
    for(int step=0; step<STEPS; step++)
    {
        // Copy my data to send buffer (Parallelized copy)
        #pragma omp parallel for schedule(static)
        for(int i=0; i<my_N; i++) {
            send_x[i] = x[i]; send_y[i] = y[i]; send_z[i] = z[i]; send_m[i] = m[i];
        }

        int current_send_count = my_N;
        int src = (rank - 1 + size) % size;
        int dst = (rank + 1) % size;

        // --- Ring Loop ---
        for(int r=0; r<size; r++)
        {
            // Determine pointers for interaction
            double *ox, *oy, *oz, *om;
            int other_N;

            // MPI Communication Phase (Single Threaded - Funneled)
            if(r > 0) {
                int incoming_rank = (rank - r + size) % size;
                int incoming_count = all_counts[incoming_rank];

                MPI_Sendrecv(send_x, current_send_count, MPI_DOUBLE, dst, 0,
                             recv_x, incoming_count,     MPI_DOUBLE, src, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(send_y, current_send_count, MPI_DOUBLE, dst, 1,
                             recv_y, incoming_count,     MPI_DOUBLE, src, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(send_z, current_send_count, MPI_DOUBLE, dst, 2,
                             recv_z, incoming_count,     MPI_DOUBLE, src, 2,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv(send_m, current_send_count, MPI_DOUBLE, dst, 3,
                             recv_m, incoming_count,     MPI_DOUBLE, src, 3,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                std::swap(send_x, recv_x);
                std::swap(send_y, recv_y);
                std::swap(send_z, recv_z);
                std::swap(send_m, recv_m);
                
                other_N = incoming_count;
                current_send_count = incoming_count;
                
                ox = send_x; oy = send_y; oz = send_z; om = send_m;
            } else {
                ox = x; oy = y; oz = z; om = m;
                other_N = my_N;
            }

            // --- OpenMP + AVX-512 Computation Phase ---
            // Threads process 'my_N' particles in chunks (static schedule)
            // All threads READ from 'ox' (Other Particles) which is in shared memory.
            #pragma omp parallel
            {
                const __m512d vSoft = _mm512_set1_pd(SOFT);
                const __m512d v1p5  = _mm512_set1_pd(1.5);
                const __m512d vHalf = _mm512_set1_pd(0.5);
                const __m512d v_G   = _mm512_set1_pd(G); // Optimized G multiply

                // Schedule Static: Ensures same thread handles same 'i' every time (Cache Locality)
                #pragma omp for schedule(static)
                for(int i=0; i<my_N; i++)
                {
                    __m512d v_xi  = _mm512_set1_pd(x[i]);
                    __m512d v_yi  = _mm512_set1_pd(y[i]);
                    __m512d v_zi  = _mm512_set1_pd(z[i]);
                    __m512d v_mi  = _mm512_set1_pd(m[i]); // We need G*mi later

                    __m512d fx0=_mm512_setzero_pd(), fy0=_mm512_setzero_pd(), fz0=_mm512_setzero_pd();
                    __m512d fx1=_mm512_setzero_pd(), fy1=_mm512_setzero_pd(), fz1=_mm512_setzero_pd();

                    int j = 0;
                    for(; j <= other_N - 16; j+=16)
                    {
                        // Prefetch other particles
                        _mm_prefetch((char*)&ox[j+64], _MM_HINT_T0);
                        _mm_prefetch((char*)&oy[j+64], _MM_HINT_T0);
                        _mm_prefetch((char*)&oz[j+64], _MM_HINT_T0);
                        _mm_prefetch((char*)&om[j+64], _MM_HINT_T0);

                        __m512d x0=_mm512_load_pd(&ox[j]);   __m512d x1=_mm512_load_pd(&ox[j+8]);
                        __m512d y0=_mm512_load_pd(&oy[j]);   __m512d y1=_mm512_load_pd(&oy[j+8]);
                        __m512d z0=_mm512_load_pd(&oz[j]);   __m512d z1=_mm512_load_pd(&oz[j+8]);
                        __m512d m0=_mm512_load_pd(&om[j]);   __m512d m1=_mm512_load_pd(&om[j+8]);

                        __m512d dx0=_mm512_sub_pd(x0,v_xi); __m512d dx1=_mm512_sub_pd(x1,v_xi);
                        __m512d dy0=_mm512_sub_pd(y0,v_yi); __m512d dy1=_mm512_sub_pd(y1,v_yi);
                        __m512d dz0=_mm512_sub_pd(z0,v_zi); __m512d dz1=_mm512_sub_pd(z1,v_zi);

                        __m512d r20 = _mm512_fmadd_pd(dx0,dx0,vSoft);
                        r20 = _mm512_fmadd_pd(dy0,dy0,r20);
                        r20 = _mm512_fmadd_pd(dz0,dz0,r20);

                        __m512d r21 = _mm512_fmadd_pd(dx1,dx1,vSoft);
                        r21 = _mm512_fmadd_pd(dy1,dy1,r21);
                        r21 = _mm512_fmadd_pd(dz1,dz1,r21);

                        __m512d inv0 = _mm512_rsqrt14_pd(r20);
                        __m512d inv1 = _mm512_rsqrt14_pd(r21);

                        __m512d t0 = _mm512_mul_pd(r20, _mm512_mul_pd(inv0, inv0));
                        t0 = _mm512_fnmadd_pd(vHalf, t0, v1p5); 
                        inv0 = _mm512_mul_pd(inv0, t0);

                        __m512d t1 = _mm512_mul_pd(r21, _mm512_mul_pd(inv1, inv1));
                        t1 = _mm512_fnmadd_pd(vHalf, t1, v1p5);
                        inv1 = _mm512_mul_pd(inv1, t1);

                        __m512d inv30 = _mm512_mul_pd(inv0,_mm512_mul_pd(inv0,inv0));
                        __m512d inv31 = _mm512_mul_pd(inv1,_mm512_mul_pd(inv1,inv1));
                        
                        // F = G * mi * mj * inv3
                        // Precompute G*mi outside loop? No, m[i] varies per thread/i.
                        // Optimization: Combine G*mi
                        __m512d v_Gmi = _mm512_mul_pd(v_G, v_mi);

                        __m512d f0 = _mm512_mul_pd(_mm512_mul_pd(v_Gmi,m0),inv30);
                        __m512d f1 = _mm512_mul_pd(_mm512_mul_pd(v_Gmi,m1),inv31);

                        fx0 = _mm512_fmadd_pd(f0,dx0,fx0); fx1 = _mm512_fmadd_pd(f1,dx1,fx1);
                        fy0 = _mm512_fmadd_pd(f0,dy0,fy0); fy1 = _mm512_fmadd_pd(f1,dy1,fy1);
                        fz0 = _mm512_fmadd_pd(f0,dz0,fz0); fz1 = _mm512_fmadd_pd(f1,dz1,fz1);
                    }

                    double fx = _mm512_reduce_add_pd(_mm512_add_pd(fx0,fx1));
                    double fy = _mm512_reduce_add_pd(_mm512_add_pd(fy0,fy1));
                    double fz = _mm512_reduce_add_pd(_mm512_add_pd(fz0,fz1));

                    for(; j < other_N; j++){
                        if (r==0 && i==j) continue; 
                        double dx=ox[j]-x[i], dy=oy[j]-y[i], dz=oz[j]-z[i];
                        double r2=dx*dx+dy*dy+dz*dz+SOFT;
                        double inv=1.0/sqrt(r2);
                        double inv3=inv*inv*inv;
                        double f=G*m[i]*om[j]*inv3;
                        fx+=f*dx; fy+=f*dy; fz+=f*dz;
                    }

                    double invMdt = DT/m[i];
                    vx[i] += fx * invMdt;
                    vy[i] += fy * invMdt;
                    vz[i] += fz * invMdt;
                }
            } // End OpenMP Parallel
        } // End Ring Loop

        // --- Position Update (Parallelized) ---
        #pragma omp parallel
        {
            const __m512d vDt = _mm512_set1_pd(DT);
            int limit = my_N - (my_N % 8);
            
            #pragma omp for schedule(static)
            for(int i=0; i < limit; i+=8) {
                __m512d v_x  = _mm512_load_pd(&x[i]);
                __m512d v_y  = _mm512_load_pd(&y[i]);
                __m512d v_z  = _mm512_load_pd(&z[i]);
                __m512d v_vx = _mm512_load_pd(&vx[i]);
                __m512d v_vy = _mm512_load_pd(&vy[i]);
                __m512d v_vz = _mm512_load_pd(&vz[i]);

                _mm512_store_pd(&x[i], _mm512_fmadd_pd(v_vx, vDt, v_x));
                _mm512_store_pd(&y[i], _mm512_fmadd_pd(v_vy, vDt, v_y));
                _mm512_store_pd(&z[i], _mm512_fmadd_pd(v_vz, vDt, v_z));
            }
            #pragma omp single
            {
                for(int i=limit; i<my_N; i++) {
                    x[i]+=vx[i]*DT;
                    y[i]+=vy[i]*DT;
                    z[i]+=vz[i]*DT;
                }
            }
        }

        // --- Visualization Gathering (Only if active) ---
        if(mode==VISUAL && step%saveInterval==0){
             // Gathering is complex in hybrid, usually handled by master thread of each rank
             // and then MPI_Gatherv.
             vector<double> g_x, g_y, g_z, g_vx, g_vy, g_vz, g_m;
             if(rank == 0) {
                 g_x.resize(N_GLOBAL); g_y.resize(N_GLOBAL); g_z.resize(N_GLOBAL);
                 g_vx.resize(N_GLOBAL); g_vy.resize(N_GLOBAL); g_vz.resize(N_GLOBAL); g_m.resize(N_GLOBAL);
             }
             MPI_Gatherv(x, my_N, MPI_DOUBLE, g_x.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
             MPI_Gatherv(y, my_N, MPI_DOUBLE, g_y.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
             MPI_Gatherv(z, my_N, MPI_DOUBLE, g_z.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
             MPI_Gatherv(vx, my_N, MPI_DOUBLE, g_vx.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
             MPI_Gatherv(vy, my_N, MPI_DOUBLE, g_vy.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
             MPI_Gatherv(vz, my_N, MPI_DOUBLE, g_vz.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
             MPI_Gatherv(m, my_N, MPI_DOUBLE, g_m.data(), all_counts.data(), all_displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
             
             if(rank==0) {
                for(int k=0; k<N_GLOBAL; k++) 
                    out<<step<<","<<k<<","<<g_x[k]<<","<<g_y[k]<<","<<g_z[k]<<","
                       <<g_vx[k]<<","<<g_vy[k]<<","<<g_vz[k]<<","<<g_m[k]<<"\n";
             }
        }
    }

    auto end=chrono::high_resolution_clock::now();
    double t=chrono::duration<double>(end-start).count();
    double max_t;
    MPI_Reduce(&t, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        double ops=(double)N_GLOBAL*N_GLOBAL*20*STEPS;
        cout<<"Time: "<<max_t<<" s\n";
        cout<<"GFLOPs: "<<(ops/max_t)/1e9<<"\n";
    }

    MPI_Finalize();
}
