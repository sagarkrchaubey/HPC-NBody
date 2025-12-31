#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <omp.h>
#include <cstring> 

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

    #pragma omp parallel for schedule(static)
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
    std::string mode = "bench";
    int save_interval = 1;

    if (argc >= 2) n_bodies = std::atoi(argv[1]);
    if (argc >= 3) n_steps = std::atoi(argv[2]);
    if (argc >= 4) mode = std::string(argv[3]);
    if (argc >= 5 && mode == "visual") save_interval = std::atoi(argv[4]);

    std::vector<Body> bodies(n_bodies);
    initialize_bodies(bodies);

    std::ofstream outfile;
    if (mode == "visual") {
        std::string filename = "nbody_output_openmp_N" + std::to_string(n_bodies) + ".csv";
        outfile.open(filename.c_str());
        outfile << "step,i,x,y,z,vx,vy,vz,m" << std::endl;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < n_steps; step++) {
        if (mode == "visual" && step % save_interval == 0) {
            for (int i = 0; i < n_bodies; i++) {
                outfile << step << "," << i << "," 
                        << bodies[i].x << "," << bodies[i].y << "," << bodies[i].z << ","
                        << bodies[i].vx << "," << bodies[i].vy << "," << bodies[i].vz << ","
                        << bodies[i].mass << "\n";
            }
        }
        update_physics(bodies);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    double total_wall_time = elapsed.count();
    
    if (mode == "visual") outfile.close();

    double total_ops = (double)n_bodies * n_bodies * 20.0 * n_steps;
    double gflops = (total_ops / total_wall_time) / 1e9;

    std::cout << "Time:   " << total_wall_time << " s" << std::endl;
    std::cout << "GFLOPs: " << gflops << "\n";

    return 0;
}
