#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    unsigned int seed = world_rank;
    long long int toss_per_process = tosses / world_size;
    long long int total = 0;
    long long int number_in_circle = 0;
    bool sended = false;
    
    for (long long int toss = 0; toss < toss_per_process; toss++) {
        double x = (double)rand_r(&seed)/RAND_MAX * 2 - 1;
        double y = (double)rand_r(&seed)/RAND_MAX * 2 - 1;

        double distance_squared = (x * x) + (y * y);
        if (distance_squared <= 1) 
            number_in_circle++;        
    }

    // TODO: binary tree redunction
    int base = 2;
    while (base <= world_size) {
        if (world_rank % base == 0) {
            // receiver
            MPI_Status status; // for MPI_recv
            total += number_in_circle;
            long long int recv_cnt;
            MPI_Recv(&recv_cnt, 1, MPI_LONG_LONG_INT, world_rank+(base>>1), 0, MPI_COMM_WORLD, &status);
            total += recv_cnt;
        }else {
            // sender
            if (!sended) {
                MPI_Send(&number_in_circle, 1, MPI_LONG_LONG_INT, world_rank-(base>>1), 0, MPI_COMM_WORLD);
                sended = true;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        base *= 2;
    } 
    

    if (world_rank == 0)
    {
        // TODO: PI result
        pi_result = 4 * total / (double)tosses;
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
