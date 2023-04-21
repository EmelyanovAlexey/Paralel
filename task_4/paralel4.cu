/*
 * Реализация уравнения теплопроводности в двумерной области
 * на равномерных сетках с использованием CUDA.
 */

#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <string>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// углы нашей матрицы
constexpr double ANGLE1 = 10;
constexpr double ANGLE2 = 20;
constexpr double ANGLE3 = 30;
constexpr double ANGLE4 = 20;

// функция для переращета нашей матрицы уравнения теплопроводности
__global__ void calculate(double *arr, double *newArr)
{
    size_t i = blockIdx.x;
    size_t j = threadIdx.x;
    size_t size = blockDim.x + 1;

    if (i != 0 && i != size - 1 && j != 0 && j != size - 1)
    {
        newArr[i * size + j] = 0.25 * (arr[i * size + j - 1] + arr[(i - 1) * size + j] +
                                       arr[(i + 1) * size + j] + arr[i * size + j + 1]);
    }
}

// функция для получение разнец между массивами
__global__ void getDifference(double *arr, double *newArr, double *difArr)
{
    int blockIndex = blockIdx.x + gridDim.y * blockIdx.y;
    int threadIndex = threadIdx.x + threadIdx.y * blockDim.x;

    int arrayIndex = blockIndex * blockDim.x * blockDim.y + threadIndex;
    int gridX = gridDim.x * blockDim.x;
    int gridY = gridDim.y * blockDim.y;
    int i = arrayIndex / gridX;
    int j = arrayIndex % gridY;
    if (i != 0 && i != gridY - 1 && j != 0 && j != gridX - 1)
    {
        difArr[i * gridX + j] = std::abs(arr[i * gridX + j] - newArr[i * gridX + j]);
    }
}

int main(int argc, char *argv[])
{
    int n = 0;             // размер
    int MAX_ITERATION = 0; // максимальное число итераций
    double ACCURACY = 0;   // точность
    double error = 1.0;    // ошибка
    int cntIteration = 0;  // количество итераций
    size_t tempStorageBytes = 0;
    double *tempStorage = NULL;

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

    double *arr = new double[n * n];
    double *arrNew = new double[n * n];

    // инициализация масива и копии массива
    for (int i = 1; i < n * n; i++)
    {
        arr[i] = 20;
        arrNew[i] = 20;
    }

    // заполнение краев рамок нашей матрицы и матрицы клона
    arr[0] = ANGLE1;
    arr[n - 1] = ANGLE2;
    arr[n * (n - 1)] = ANGLE4;
    arr[n * n - 1] = ANGLE3;
    arrNew[0] = ANGLE1;
    arrNew[n - 1] = ANGLE2;
    arrNew[n * (n - 1)] = ANGLE4;
    arrNew[n * n - 1] = ANGLE3;

    // заполнение рамок нашей матрицы и матрицы клона
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

    // выделяем память на gpu через cuda для 3 сеток
    double *CudaArr, *CudaNewArr, *CudaDifArr;
    cudaMalloc((void **)&CudaArr, sizeof(double) * n * n);
    cudaMalloc((void **)&CudaNewArr, sizeof(double) * n * n);
    cudaMalloc((void **)&CudaDifArr, sizeof(double) * n * n);

    // копирование информации с CPU на GPU
    cudaMemcpy(CudaArr, arr, sizeof(double) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(CudaNewArr, arrNew, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    // выделяем память на gpu. Хранение ошибки на device
    double *maxError = 0;
    cudaMalloc((void **)&maxError, sizeof(double));

    // получаем размер временного буфера для редукции
    cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaDifArr, maxError, n * n);

    // выделяем память для буфера
    cudaMalloc((void **)&tempStorage, tempStorageBytes);

    // размерность блоков и грида
    dim3 blockDim = dim3(32, 32);
    std::cout << blockDim.x << blockDim.y << blockDim.z << std::endl;
    dim3 gridDim = dim3((n+blockDim.x-1)/blockDim.x,(n+blockDim.y-1)/blockDim.y);

    while (cntIteration < MAX_ITERATION && error > ACCURACY)
    {
        cntIteration++;
        calculate<<<n - 1, n - 1>>>(CudaArr, CudaNewArr); // расчет матрицы

        // рaсчитываем ошибку
        if (cntIteration % 100 == 0)
        {
            // вычисления разницы
            getDifference<<<gridDim, blockDim>>>(CudaArr, CudaNewArr, CudaDifArr);
            // нахождение максимума в разнице матрицы
            cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaDifArr, maxError, n * n);
            // запись ошибки в переменную
            cudaMemcpy(&error, maxError, sizeof(double), cudaMemcpyDeviceToHost);
            error = std::abs(error);
        }

        double *copy = arr;
        arr = arrNew;
        arrNew = copy;
    }

    // вывод резуьтата
    std::cout << "iteration: " << cntIteration << " \n"
              << "error: " << error << "\n";

    // чистка памяти
    cudaFree(CudaArr);
    cudaFree(CudaNewArr);
    cudaFree(CudaDifArr);
    delete[] arr;
    delete[] arrNew;
    return 0;
}
