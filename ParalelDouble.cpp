#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <time.h>

const unsigned int N = 10000000;
double arr[N];
double pi = 3.1415926535;
double x;

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    double sum = 0;
    auto startCreate = std::chrono::high_resolution_clock::now();
#pragma acc enter data create(arr [0:N], sum)

#pragma acc parallel loop present(arr [0:N])
    for (unsigned int i = 0; i < N; i++)
    {
        arr[i] = sin(2 * pi * i / N);
    }
    auto elapsed = std::chrono::high_resolution_clock::now() - startCreate;
    long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
                            elapsed)
                            .count();
    std::cout << microseconds << " create time\n";

    auto startSum = std::chrono::high_resolution_clock::now();
#pragma acc parallel loop present(arr [0:N], sum) reduction(+ \
                                                            : sum)
    for (unsigned int i = 0; i < N; i++)
    {
        sum += arr[i];
    }
#pragma acc exit data delete (arr [0:N])copyout(sum)
    elapsed = std::chrono::high_resolution_clock::now() - startSum;
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
                       elapsed)
                       .count();
    std::cout << microseconds << " sum time\n";

    elapsed = std::chrono::high_resolution_clock::now() - start;
    microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
                       elapsed)
                       .count();

    std::cout << microseconds << " microseconds\n";

    std::cout << sum << " sum\n";
    return 0;
}