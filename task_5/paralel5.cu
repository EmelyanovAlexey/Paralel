/* 
 * Реализация уравнения теплопроводности в двумерной области
 * на равномерных сетках с использованием CUDA. 
 * Операция редукции (вычисление максимального значения ошибки)
 * для одного MPI процесса реализуется с помощью библиотеки CUB.
 * Подсчет глобального значения ошибки, обмен граничными условиями
 * реализуется с использованием MPI 
*/

// подключение библиотек
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mpi.h>

using namespace std;

// функция, обновляющая граничные значения сетки
__global__ void update_boundaries(double* arr, double* arrNew, size_t n, size_t sizePerGpu)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, находится ли текущий поток в границах массива
	if(i <= n - 2 && i > 0){
        // Обновляем граничный элемент arrNew[1 * n + i - 1], используя среднее значение его соседних элементов в массиве arr
		arrNew[1 * n + i] = 0.25 * (arr[1 * n + i - 1] + arr[(1 - 1) * n + i] + arr[(1 + 1) * n + i] + arr[1 * n + i + 1]);
		arrNew[(sizePerGpu - 2) * n + j] = 0.25 * (arr[(sizePerGpu - 2) * n + j - 1] + arr[((sizePerGpu - 2) - 1) * n + j] + arr[((sizePerGpu - 2) + 1) * n + j] + arr[(sizePerGpu - 2) * n + j + 1]);
	}
}

// функция, обновляющая внутренние значения сетки
__global__ void update(double* arr, double* arrNew, int n, int sizePerGpu)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Проверяем, находятся ли индексы j и i внутри границ сетки
	if(j >= 1 && i >= 2 && j <= n - 2 && i <= sizePerGpu - 2){
        // Вычисляем новое значение элемента в arrNew по среднему значению его соседей
		arrNew[i*n + j] = 0.25 * (arr[i * n + j - 1] + arr[i * n + j + 1] + arr[(i - 1) * n + j] + arr[(i + 1) * n + j]);
	}
}

// функция нахождения разности двух массивов
__global__ void substract(double* arr, double* arrNew, double* res, int n, int sizePerGpu){
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;

    // Проверяем, находятся ли индексы j и i внутри границ массивов
	if(j > 0 && i > 0 && j < n - 1 && i < sizePerGpu - 1){
        // Вычисляем разность соответствующих элементов arrNew и arr и записываем результат в res
		res[i * n + j] = arrNew[i * n + j] - arr[i * n + j];
	}
}

// основное тело программы
int main(int argc, char* argv[]){
    // Объявление и инициализация переменных myRank и nRanks для определения ранга и общего числа процессов в MPI.
    int myRank, nRanks;
    // Инициализация MPI.
    MPI_Init(&argc, &argv);
    // Получение ранга текущего процесса.
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    // Получение общего числа процессов.
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    // Проверка условия, что число процессов должно быть от 1 до 2.
    if(nRanks < 1 || nRanks > 2) {
        printf("1-2");
        exit(0);
    }
    // Установка текущего устройства CUDA, соответствующего рангу процесса.
	cudaSetDevice(myRank);

	// инициализация переменных
    double ACCURACY;
    const int n, MAX_ITERATION;

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

	double *arr= NULL, *arrNew = NULL, *CudaArr = NULL, *CudaNewArr = NULL, *CudaDifArr;

	int cntIteration = 0; 
	double dt = 10.0 / (n - 1), error = 1;
	size_t sizeForOne = n / nRanks, start_index = n / nRanks * myRank;
    cudaMemcpyToSymbol(add, &dt, sizeof(double));
    cudaMallocHost(&arr, sizeof(double) * n * n);
    cudaMallocHost(&arrNew, sizeof(double) * n * n);
    // заполняем сетку
	// fill(arr,arrNew,n);

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

	// выделяем необходимую для процесса память на GPU
	if(nRanks!=1){
		if (myRank != 0 && myRank != nRanks - 1) sizeForOne += 2;
		else sizeForOne += 1;
	 }
	cudaMalloc((void**)&CudaArr, n * sizeForOne * sizeof(double));
	cudaMalloc((void**)&CudaNewArr, n * sizeForOne * sizeof(double));
	cudaMalloc((void**)&CudaDifArr, n * sizeForOne * sizeof(double));

    // копирование информации с CPU на GPU
	size_t offset = (myRank != 0) ? n : 0;
 	cudaMemcpy(CudaArr, (arr + (start_index * n) - offset), sizeof(double) * n * sizeForOne, cudaMemcpyHostToDevice);
	cudaMemcpy(CudaNewArr, arrNew + (start_index * n) - offset, sizeof(double) * n * sizeForOne, cudaMemcpyHostToDevice);

	// создаем потоки и назначаем приоритет
	int leastPriority = 0;
    int greatestPriority = 0;
    // Получаем диапазон приоритетов потоков устройства CUDA
	cudaStream_t stream_boundaries, stream_inner;
	cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    // Создаем поток stream_boundaries с наивысшим приоритетом
	cudaStreamCreateWithPriority(&stream_boundaries, cudaStreamDefault, greatestPriority);
    // Создаем поток stream_inner с наименьшим приоритетом
	cudaStreamCreateWithPriority(&stream_inner, cudaStreamDefault, leastPriority);

    dim3 threadPerBlock(32, 32);
    dim3 blocksPerGrid(n / ((n<1024)? n:1024), sizeForOne);

	double* d_error;
    cudaMalloc(&d_error, sizeof(double));

	// определяем требования к временному хранилищу устройства и выделяем память
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, CudaDifArr, d_error, n*n);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// цикл пересчета ошибки и обновления сетки 
	while((error > ACCURACY) && (cntIteration < MAX_ITERATION)){
		cntIteration++;
		
		// расчет границ
		update_boundaries<<<n, 1, 0, stream_boundaries>>>(CudaArr, CudaNewArr, n, sizeForOne);
		cudaStreamSynchronize(stream_boundaries);

		// расчет внутренних значений
		update<<<threadPerBlock, blocksPerGrid, 0, stream_inner>>>(CudaArr, CudaNewArr, n, sizeForOne);

		// пересчет значения ошибки раз в 200 итераций
		if (cntIteration % 200 == 0){
			substract<<<threadPerBlock, blocksPerGrid, 0, stream_inner>>>(CudaArr, CudaNewArr, CudaDifArr, n, sizeForOne);
			// Результат сохраняется в d_error, выделенной памяти d_temp_storage, и размере 
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, CudaDifArr, d_error, n * sizeForOne, stream_inner);
			// Синхронизируем поток stream_inner, чтобы убедиться, что все операции завершены
            cudaStreamSynchronize(stream_inner);
            // Выполняем операцию MPI_Allreduce для получения максимального значения ошибки
			MPI_Allreduce((void*)d_error, (void*)d_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            // Асинхронно копируем значение ошибки из устройства в хост
			cudaMemcpyAsync(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost, stream_inner);
		}
		
		// верхняя граница
        // Проверяем, если текущий процесс не является первым (myRank != 0), то отправляем верхнюю границу массива CudaNewArr 
        // на предыдущий процесс и одновременно принимаем верхнюю границу от предыдущего процесса.
        	if (myRank != 0){
		    MPI_Sendrecv(CudaNewArr + n + 1, n - 2, MPI_DOUBLE, myRank - 1, 0, 
				CudaNewArr + 1, n - 2, MPI_DOUBLE, myRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// нижняя граница
        // Проверяем, если текущий процесс не является последним (myRank != nRanks - 1), то отправляем нижнюю границу массива CudaNewArr 
        // на следующий процесс и одновременно принимаем нижнюю границу от следующего процесса.
		if (myRank != nRanks - 1){
		    MPI_Sendrecv(CudaNewArr + (sizeForOne - 2) * n + 1, n - 2, MPI_DOUBLE, myRank + 1, 0,
					CudaNewArr + (sizeForOne - 1) * n + 1, 
					n - 2, MPI_DOUBLE, myRank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
        // Синхронизируем поток stream_inner, чтобы убедиться, что все операции завершены
		cudaStreamSynchronize(stream_inner);

		// обмен значениями
		double* copy = CudaArr;
		CudaArr = CudaNewArr;
		CudaNewArr = copy;
	}

    // освобождаем память
    cudaFree(CudaArr);
    cudaFree(CudaNewArr);
    cudaFree(d_error);
    cudaFree(CudaDifArr);
    cudaFree(d_error);
    cudaFree(d_temp_storage);
    MPI_Finalize();

	return 0;
}