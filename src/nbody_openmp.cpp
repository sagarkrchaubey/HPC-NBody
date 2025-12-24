#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <omp.h>
#include <sched.h> 
#include <set>     
#include <iomanip> 

const double DT = 0.001;
const double G = 1.0;
const double SOFTENING = 0.1;
const int SEED = 1234;

std::vector<double> global_thread_times;
std::vector<long long> global_thread_ops;
std::vector<int> global_core_ids;

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

void initialize_bodies(std::vector<Body>& bodies) {
    std::srand(SEED);
    for (size_t i = 0; i < bodies.size(); i++) {
        bodies[i].x = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        bodies[i].y = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        bodies[i].z = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        bodies[i].vx = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].vy = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].vz = ((double)rand() / RAND_MAX) - 0.5;
        bodies[i].mass = ((double)rand() / RAND_MAX) * 0.9 + 0.1;
    }
}

void update_physics(std::vector<Body>& bodies) {
    int N = bodies.size();
    std::vector<double> fx(N, 0.0), fy(N, 0.0), fz(N, 0.0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        global_core_ids[tid] = sched_getcpu();

        double t_start = omp_get_wtime();
        long long ops_counter = 0;

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;

                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                
                double distSq = dx*dx + dy*dy + dz*dz + SOFTENING;
                double dist = std::sqrt(distSq);
                double f = (G * bodies[i].mass * bodies[j].mass) / (dist * dist * dist);

                fx[i] += f * dx;
                fy[i] += f * dy;
                fz[i] += f * dz;
            }
            ops_counter += (N - 1); 
        }

        double t_end = omp_get_wtime();
        
        #pragma omp atomic
        global_thread_times[tid] += (t_end - t_start);
        
        #pragma omp atomic
        global_thread_ops[tid] += ops_counter;
    } 

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        double ax = fx[i] / bodies[i].mass;
        double ay = fy[i] / bodies[i].mass;
        double az = fz[i] / bodies[i].mass;

        bodies[i].vx += ax * DT;
        bodies[i].vy += ay * DT;
        bodies[i].vz += az * DT;

        bodies[i].x += bodies[i].vx * DT;
        bodies[i].y += bodies[i].vy * DT;
        bodies[i].z += bodies[i].vz * DT;
    }
}

int main(int argc, char* argv[]) {
    int n_bodies = 100;
    int n_steps = 100;
    bool benchmark_mode = false; 

    if (argc >= 2) n_bodies = std::atoi(argv[1]);
    if (argc >= 3) n_steps = std::atoi(argv[2]);
    if (argc >= 4 && std::string(argv[3]) == "true") benchmark_mode = true;

    int max_threads = omp_get_max_threads();
    
    global_thread_times.resize(max_threads, 0.0);
    global_thread_ops.resize(max_threads, 0);
    global_core_ids.resize(max_threads, -1);

    std::string filename = "nbody_output_openmp_N" + std::to_string(n_bodies) 
                         + "_S" + std::to_string(n_steps) 
                         + "_T" + std::to_string(max_threads) + ".csv";

    std::cout << "\n--- N-Body OpenMP Simulation ---" << std::endl;
    std::cout << "Bodies: " << n_bodies << " | Steps: " << n_steps << std::endl;
    std::cout << "Threads: " << max_threads << std::endl;
    std::cout << "Output File: " << filename << std::endl;

    std::vector<Body> bodies(n_bodies);
    initialize_bodies(bodies);

    std::ofstream outfile;
    if (!benchmark_mode) {
        outfile.open(filename.c_str());
        outfile << "Step,BodyID,X,Y,Z,Mass" << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < n_steps; step++) {
        if (!benchmark_mode && step % 1 == 0) {
            for (int i = 0; i < n_bodies; i++) {
                outfile << step << "," << i << "," 
                        << bodies[i].x << "," << bodies[i].y << "," << bodies[i].z 
                        << "," << bodies[i].mass << "\n";
            }
        }
        update_physics(bodies);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double total_wall_time = elapsed.count();

    if (!benchmark_mode) outfile.close();

    std::cout << "\n========================================================" << std::endl;
    std::cout << "                PERFORMANCE REPORT                      " << std::endl;
    std::cout << "========================================================" << std::endl;
    
    std::set<int> unique_cores;
    for(int id : global_core_ids) if(id != -1) unique_cores.insert(id);
    
    std::cout << "Threads Active:   " << max_threads << std::endl;
    std::cout << "Unique Cores Used: " << unique_cores.size() << std::endl;

    std::cout << "\n--- Per-Thread Breakdown ---" << std::endl;
    std::cout << std::left << std::setw(10) << "ThreadID" 
              << std::setw(15) << "Compute Time(s)" 
              << std::setw(20) << "Interactions(Ops)" 
              << std::setw(15) << "Est. GFLOPs" << std::endl;

    double total_parallel_ops = 0;
    for (int t = 0; t < max_threads; t++) {
        double thread_gflops = 0.0;
        if (global_thread_times[t] > 0) {
             thread_gflops = (global_thread_ops[t] * 20.0) / global_thread_times[t] / 1e9;
        }
        
        std::cout << std::left << std::setw(10) << t 
                  << std::setw(15) << std::fixed << std::setprecision(4) << global_thread_times[t] 
                  << std::setw(20) << global_thread_ops[t]
                  << std::setw(15) << thread_gflops << std::endl;
        
        total_parallel_ops += (global_thread_ops[t] * 20.0);
    }

    double combined_gflops = (total_parallel_ops / total_wall_time) / 1e9;
    
    std::cout << "\n--- Overall Summary ---" << std::endl;
    std::cout << "Wall Clock Time:   " << total_wall_time << " s" << std::endl;
    std::cout << "Combined GFLOPs:   " << combined_gflops << " GFLOPs" << std::endl;
    std::cout << "========================================================" << std::endl;

    return 0;
}
