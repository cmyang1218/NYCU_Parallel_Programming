#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <pthread.h>

long long int num_in_circles = 0;

pthread_mutex_t mutex;
  
typedef struct args {
    int id;
    int num;
} args_t;

void* calculate_pi(void* args) {
    args_t* arg = (args_t *)args;

    unsigned int seed = arg->id;
    long long int num_hit = 0;
    
    for (long long int idx = 0; idx < arg->num; idx++) {
        double x = (double)rand_r(&seed)/RAND_MAX * 2 - 1;
        double y = (double)rand_r(&seed)/RAND_MAX * 2 - 1;
        double distance = x * x + y * y;
        if (distance <= 1.0f) 
            num_hit++;
    }
    pthread_mutex_lock(&mutex);
    num_in_circles += num_hit;
    pthread_mutex_unlock(&mutex);
    
    return NULL;
}

int main (int argc, char *argv[]) {
    const int num_of_thread = std::atoi(argv[1]);
    const long long int num_of_tosses = std::atoll(argv[2]);
    long long int toss_per_thread = num_of_tosses / num_of_thread;   
    
    pthread_t* threads = new pthread_t[num_of_thread];
    args_t* args = new args_t[num_of_thread];

    pthread_mutex_init(&mutex, NULL);
    long long int num;
    for (int i = 0; i < num_of_thread; i++) {
        if (i == num_of_thread-1 && num_of_tosses % num_of_thread != 0) {
            num = toss_per_thread + num_of_tosses % num_of_thread;
        }else {
            num = toss_per_thread;
        }
	args[i].id = i;
	args[i].num = num;
        pthread_create(&threads[i], NULL, calculate_pi, &args[i]);
    }
    for (int i = 0; i < num_of_thread; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_mutex_destroy(&mutex);
    double pi_estimate = 4.0f * num_in_circles / ((double)num_of_tosses);
    std::printf("%lf\n", pi_estimate);

    delete[] threads;
    return 0;
}
