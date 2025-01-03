#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

bool fnz(long long int *schedule, long long int *oldschedule, int size)
{
    int res = 0;
    for (int i = 1; i < size; i++)
    {    
        if (schedule[i] == 0) 
        {
            res++;
        }
        else if (schedule[i] != oldschedule[i]) 
        {
            oldschedule[i] = schedule[i];
        }
    }
    // return true if all schedules of worker are modified
    return(res == 0);
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;
    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    unsigned int seed = world_rank;   
    long long int toss_per_process = tosses / world_size;
    long long int total = 0;
    long long int number_in_circle = 0;

    for (long long int toss = 0; toss < toss_per_process; toss++) {
        double x = (double)rand_r(&seed)/RAND_MAX * 2 - 1;
        double y = (double)rand_r(&seed)/RAND_MAX * 2 - 1;

        double distance_squared = (x * x) + (y * y);
        if (distance_squared <= 1) 
            number_in_circle++;
    }

    if (world_rank == 0)
    {
        // Master
        long long int *oldschedule = (long long int *)malloc(world_size * sizeof(long long int));
        long long int *schedule;
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &schedule);

        for (int i = 0; i < world_size; i++) {
            schedule[i] = 0;
            oldschedule[i] = -1;
        }

        MPI_Win_create(schedule, world_size * sizeof(long long int), sizeof(long long int), MPI_INFO_NULL,
             MPI_COMM_WORLD, &win);
        
        bool ready = 0;
        while (!ready) {
            MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
            ready = fnz(schedule, oldschedule, world_size);
            MPI_Win_unlock(0, win);
        }
        total = number_in_circle;
        for (int p = 1; p < world_size; p++) {
            total += schedule[p];
        }

        MPI_Free_mem(schedule);
        free(oldschedule);
    }
    else
    {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Put(&number_in_circle, 1, MPI_LONG_LONG_INT, 0, 
             world_rank, 1, MPI_LONG_LONG_INT, win);
        MPI_Win_unlock(0, win);
    }

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
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