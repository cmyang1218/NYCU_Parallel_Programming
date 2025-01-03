#include <iostream>
#include <cmath>


int main (int argc, char* argv[]) {
    int number_in_circle = 0;
    int number_of_tosses = std::atoi(argv[1]);
    srand(time(NULL));

    for (int toss = 0; toss < number_of_tosses; toss++) {
        double x = (double)rand()/RAND_MAX * 2 - 1;
        double y = (double)rand()/RAND_MAX * 2 - 1;

        double distance_squared = (x * x) + (y * y);
        if (distance_squared <= 1) 
            number_in_circle++;
    }

    double pi_estimate = 4 * number_in_circle / ((double) number_of_tosses);
    std::cout << pi_estimate << '\n';

    return 0;
}
