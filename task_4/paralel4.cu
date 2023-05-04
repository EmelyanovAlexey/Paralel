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

// функция, обновляющая значения сетки
__global__ void calculate(double *A, double *Anew, int size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < size - 1 && j > 0 && i > 0 && i < size - 1)
    {
        double left = A[i * size + j - 1];
        double right = A[i * size + j + 1];
        double top = A[(i - 1) * size + j];
        double bottom = A[(i + 1) * size + j];
        Anew[i * size + j] = 0.25 * (left + right + top + bottom);
    }
}

// функция нахождения разности двух массивов
__global__ void getDifference(double *A, double *Anew, double *res, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 0 && i < size && j >= 0 && j < size)
        res[i * size + j] = Anew[i * size + j] - A[i * size + j];
}

int main(int argc, char *argv[])
{
    int n = 0;                   // размер
    int MAX_ITERATION = 0;       // максимальное число итераций
    double ACCURACY = 0;         // точность
    double error = 1.0;          // ошибка
    int cntIteration = 0;        // количество итераций
    size_t tempStorageBytes = 0; // размер выделенной памяти для d_temp_storage
    double *tempStorage = NULL;  // доступное для устройства выделение временного хранилища

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
        arr[i] = 0;
        arrNew[i] = 0;
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
    dim3 gridDim = dim3((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // ---------------------
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    // ---------------------

    while (cntIteration < MAX_ITERATION && error > ACCURACY)
    {
        cntIteration++;

        // если граф не создан, то создаем с помощью захвата цикла for
        if (!graphCreated)
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            // обновление значений сетки (почему два раза - подробно описано в отчете)
            for (int i = 0; i < 100; i++)
            {
                calculate<<<gridDim, blockDim, 0, stream>>>(CudaNewArr, CudaArr, n);
                calculate<<<gridDim, blockDim, 0, stream>>>(CudaArr, CudaNewArr, n);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }

        // запускаем граф
        cudaGraphLaunch(instance, stream);

        // вычитаем один массив из другого
        getDifference<<<gridDim, blockDim, 0, stream>>>(CudaArr, CudaNewArr, CudaDifArr, n);

        // находим новое значение ошибки
        cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaDifArr, maxError, n * n, stream);
        cudaMemcpy(&error, maxError, sizeof(double), cudaMemcpyDeviceToHost);
        error = std::abs(error);

        // calculate<<<n - 1, n - 1>>>(CudaArr, CudaNewArr, n); // расчет матрицы

        // // рaсчитываем ошибку
        // if (cntIteration % 200 == 0)
        // {
        //     // вычисления разницы
        //     getDifference<<<gridDim, blockDim>>>(CudaArr, CudaNewArr, CudaDifArr, n);
        //     // нахождение максимума в разнице матрицы
        //     cub::DeviceReduce::Max(tempStorage, tempStorageBytes, CudaDifArr, maxError, n * n);
        //     // запись ошибки в переменную
        //     cudaMemcpy(&error, maxError, sizeof(double), cudaMemcpyDeviceToHost);
        //     error = std::abs(error);
        // }

        // double *copy = arr;
        // arr = arrNew;
        // arrNew = copy;
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
