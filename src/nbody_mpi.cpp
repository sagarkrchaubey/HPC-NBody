#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <mpi.h>

const double DT = 0.001;
const double G = 1.0;
const double SOFTENING = 0.1;
const int SEED = 1234;

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_bodies = 100;
    int n_steps = 1000;
    bool benchmark_mode = false;

    if (argc >= 2) n_bodies = std::atoi(argv[1]);
    if (argc >= 3) n_steps = std::atoi(argv[2]);
    if (argc >= 4 && std::string(argv[3]) == "true") benchmark_mode = true;

    int local_n = n_bodies / size;
    int start_index = rank * local_n;
    if (rank == size - 1) {
        local_n += (n_bodies % size);
    }
    
    std::vector<Body> all_bodies(n_bodies);
    initialize_bodies(all_bodies); 

    MPI_Datatype MPI_BODY_TYPE;
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_BODY_TYPE);
    MPI_Type_commit(&MPI_BODY_TYPE);

    std::ofstream outfile;
    if (rank == 0) {
        std::string filename = "nbody_output_mpi_N" + std::to_string(n_bodies) 
                             + "_S" + std::to_string(n_steps) 
                             + "_R" + std::to_string(size) + ".csv";
        
        if (!benchmark_mode) {
            outfile.open(filename.c_str());
            outfile << "Step,BodyID,X,Y,Z,Mass" << std::endl;
        }
        
        std::cout << "--- MPI Simulation Initialized ---" << std::endl;
        std::cout << "Ranks: " << size << " | Bodies: " << n_bodies << std::endl;
        if (!benchmark_mode) std::cout << "Output File: " << filename << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (int step = 0; step < n_steps; step++) {

        if (rank == 0 && !benchmark_mode && step % 1 == 0) {
            for (int i = 0; i < n_bodies; i++) {
                outfile << step << "," << i << "," 
                        << all_bodies[i].x << "," << all_bodies[i].y << "," << all_bodies[i].z 
                        << "," << all_bodies[i].mass << "\n";
            }
        }

        std::vector<double> fx(local_n, 0.0);
        std::vector<double> fy(local_n, 0.0);
        std::vector<double> fz(local_n, 0.0);

        for (int i = 0; i < local_n; i++) {
            int global_i = start_index + i;
            
            for (int j = 0; j < n_bodies; j++) {
                if (global_i == j) continue;

                double dx = all_bodies[j].x - all_bodies[global_i].x;
                double dy = all_bodies[j].y - all_bodies[global_i].y;
                double dz = all_bodies[j].z - all_bodies[global_i].z;
                
                double distSq = dx*dx + dy*dy + dz*dz + SOFTENING;
                double dist = std::sqrt(distSq);
                double f = (G * all_bodies[global_i].mass * all_bodies[j].mass) / (dist * dist * dist);

                fx[i] += f * dx;
                fy[i] += f * dy;
                fz[i] += f * dz;
            }
        }

        for (int i = 0; i < local_n; i++) {
            int global_i = start_index + i;
            
            double ax = fx[i] / all_bodies[global_i].mass;
            double ay = fy[i] / all_bodies[global_i].mass;
            double az = fz[i] / all_bodies[global_i].mass;

            all_bodies[global_i].vx += ax * DT;
            all_bodies[global_i].vy += ay * DT;
            all_bodies[global_i].vz += az * DT;

            all_bodies[global_i].x += all_bodies[global_i].vx * DT;
            all_bodies[global_i].y += all_bodies[global_i].vy * DT;
            all_bodies[global_i].z += all_bodies[global_i].vz * DT;
        }

        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                      &all_bodies[0], n_bodies/size, MPI_BODY_TYPE, 
                      MPI_COMM_WORLD);
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;
    double max_elapsed;
    
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (!benchmark_mode) outfile.close();
        
        double total_ops = (double)n_bodies * n_bodies * 20.0 * n_steps;
        double gflops = (total_ops / max_elapsed) / 1e9;

        std::cout << "\n--- MPI Simulation Results ---" << std::endl;
        std::cout << "Ranks: " << size << std::endl;
        std::cout << "Time:  " << max_elapsed << " s" << std::endl;
        std::cout << "Perf:  " << gflops << " GFLOPs" << std::endl;
    }

    MPI_Type_free(&MPI_BODY_TYPE);
    MPI_Finalize();

    return 0;
}
