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
    long long total = 0;
    long long number_in_circle = 0;

    for (long long int toss = 0; toss < toss_per_process; toss++) {
        double x = (double)rand_r(&seed)/RAND_MAX * 2 - 1;
        double y = (double)rand_r(&seed)/RAND_MAX * 2 - 1;

        double distance_squared = (x * x) + (y * y);
        if (distance_squared <= 1) 
            number_in_circle++;
    }

    if (world_rank > 0)
    {
        // TODO: MPI workers
        MPI_Request request;
        MPI_Isend(&number_in_circle, 1, MPI_LONG_LONG_INT, 0, 0, MPI_COMM_WORLD, &request);
    }
    else if (world_rank == 0)
    {
        // TODO: non-blocking MPI communication.
        // Use MPI_Irecv, MPI_Wait or MPI_Waitall.
        MPI_Request *requests = new MPI_Request[world_size-1];
        MPI_Status *status = new MPI_Status[world_size-1]; // for MPI_Irecv;
        total = number_in_circle;
        long long int *recv_cnt = new long long int[world_size-1];
        for (int p = 1; p < world_size; p++) {
            MPI_Irecv(&recv_cnt[p-1], 1, MPI_LONG_LONG_INT, p, 0, MPI_COMM_WORLD, &requests[p-1]);
        }
        MPI_Waitall(world_size-1, requests, status);
        for (int p = 0; p < world_size-1; p++) {
            total += recv_cnt[p];
        }
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
