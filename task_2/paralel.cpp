#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <stdlib.h>
#include <openacc.h>

constexpr double ANGLE1 = 10;
constexpr double ANGLE2 = 20;
constexpr double ANGLE3 = 30;
constexpr double ANGLE4 = 20;

int main(int argc, char *argv[])
{
    int n = 0;
    int MAX_ITERATION = 0;
    double ACCURACY = 0;
    double error = 1.0;
    int cntIteration = 0;

    // считываем с командной строки
    for (int arg = 1; arg < argc; arg++)
    {
        if (arg == 1)
            ACCURACY = atof(argv[arg]);
        if (arg == 2)
            n = std::stoi(argv[arg]);
        if (arg == 3)
            MAX_ITERATION = std::stoi(argv[arg]);
    }

    // определяем наши шаги по рамкам
    double dt = 10 / ((double)n - 1);
    double *arr = new double[n * n];    // работаем с данным массивом
    double *arrNew = new double[n * n]; // для записи резульата

    // init
    for (int i = 1; i < n * n; i++)
    {
        arr[i] = 20;
        arrNew[i] = 20;
    }

    // края
    arr[0] = ANGLE1;
    arr[n - 1] = ANGLE2;
    arr[n * (n - 1)] = ANGLE4;
    arr[n * n - 1] = ANGLE3;
    arrNew[0] = ANGLE1;
    arrNew[n - 1] = ANGLE2;
    arrNew[n * (n - 1)] = ANGLE4;
    arrNew[n * n - 1] = ANGLE3;

    // рамки
    for (int i = 1; i < n - 1; i++)
    {
        arr[i] = arr[i - 1] + dt;
        arr[i * n + n - 1] = arr[(i - 1) * n + n - 1] + dt;
        arr[i * n] = arr[(i - 1) * n] + dt;
        arrNew[i] = arr[i - 1] + dt;
        arrNew[i * n + n - 1] = arrNew[(i - 1) * n + n - 1] + dt;
        arrNew[i * n] = arrNew[(i - 1) * n] + dt;
    }
    for (int i = 0; i < n - 2; i++)
    {
        arr[n * (n - 1) + i + 1] = arr[n * (n - 1) + i] + dt;
        arrNew[n * (n - 1) + i + 1] = arrNew[n * (n - 1) + i] + dt;
    }

    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         std::cout << arr[(i * n) + j] << " ";
    //     }
    //     std::cout << " \n";
    // }

#pragma acc enter data copyin(error, arr [0:(n * n)], arrNew [0:(n * n)])
    {

        while (cntIteration < MAX_ITERATION && error > ACCURACY)
        {
            if (cntIteration % 10 == 0)
            {
#pragma acc kernels async(0)
                error = 0;
#pragma acc update device(error) async(0)
            }

#pragma acc data present(arrNew, arr, error)
#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256) reduction(max \
                                                                                                          : error) async(0)
            for (size_t i = 1; i < n - 1; i++)
            {
                for (size_t j = 1; j < n - 1; j++)
                {
                    arrNew[i * n + j] = 0.25 * (arr[(i + 1) * n + j] + arr[(i - 1) * n + j] + arr[i * n + j - 1] + arr[i * n + j + 1]);
                    error = fmax(error, fabs(arrNew[i * n + j] - arr[i * n + j]));
                }
            }
            if (cntIteration % 200 == 0)
            {
#pragma acc update host(error) async(0)

#pragma acc wait(0)
            }
            cntIteration++;
            double *copy = arr;
            arr = arrNew;
            arrNew = copy;
        }
    }
    std::cout << "iteration: " << cntIteration << " \n"
              << "error: " << error << "\n";
    delete[] arr;
    delete[] arrNew;
    return 0;
}
