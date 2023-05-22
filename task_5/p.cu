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
__global__ void update_boundaries(double* A, double* Anew, size_t size, size_t sizePerGpu)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if(i <= size - 2 && i > 0){
		Anew[1 * size + i] = 0.25 * (A[1 * size + i - 1] + A[(1 - 1) * size + i] + A[(1 + 1) * size + i] + A[1 * size + i + 1]);
		Anew[(sizePerGpu - 2) * size + j] = 0.25 * (A[(sizePerGpu - 2) * size + j - 1] + A[((sizePerGpu - 2) - 1) * size + j] + A[((sizePerGpu - 2) + 1) * size + j] + A[(sizePerGpu - 2) * size + j + 1]);
	}
}

// функция, обновляющая внутренние значения сетки
__global__ void update(double* A, double* Anew, int size, int sizePerGpu)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(j >= 1 && i >= 2 && j <= size - 2 && i <= sizePerGpu - 2){
		double left = A[i * size + j - 1];
		double right = A[i * size + j + 1];
		double top = A[(i - 1) * size + j];
		double bottom = A[(i + 1) * size + j];
		Anew[i*size + j] = 0.25 * (left + right + top + bottom);
	}
}

// функция нахождения разности двух массивов
__global__ void substract(double* A, double* Anew, double* res, int size, int sizePerGpu){
	unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    	unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	if(j > 0 && i > 0 && j < size - 1 && i < sizePerGpu - 1){
		res[i * size + j] = Anew[i * size + j] - A[i * size + j];
	}
}

__constant__ double add;

// функция для заполнения массивов
void fill(double* A, double* Anew, int size) 
{
    	A[0] = 10;
	A[size - 1] = 20;
	A[size * size - 1] = 30;
	A[size * (size - 1)] = 20;
	Anew[0] = 10;
	Anew[size - 1] = 20;
	Anew[size * size - 1] = 30;
	Anew[size * (size - 1)] = 20;

	const double add = 10.0 / (size - 1);
	for (int i = 1; i < size - 1; i++)
	{
		A[i] = 10 + i * add;
		A[i * size] = 10 + i * add;
		A[size - 1 + i * size] = 20 + i * add;
		A[size * (size - 1) + i] = 20 + i * add;
		Anew[i] = 10 + i * add;
		Anew[i * size] = 10 + i * add;
		Anew[size - 1 + i * size] = 20 + i * add;
		Anew[size * (size - 1) + i] = 20 + i * add;
	}
}

// основное тело программы
int main(int argc, char* argv[]){
     	// время до выполнения программы
	auto begin1 = std::chrono::steady_clock::now();

        // инициализация MPI
        int myRank, nRanks;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

        if(nRanks < 1 || nRanks > 2) {
        	printf("1-2");
        	exit(0);
        }

	cudaSetDevice(myRank);

	// инициализация переменных
        double tol = atof(argv[1]);
        const int size = atoi(argv[2]), iter_max = atoi(argv[3]);
	double *A= NULL, *Anew = NULL, *d_A = NULL, *d_Anew = NULL, *d_Asub;

	int iter = 0; 
	double addH = 10.0 / (size - 1), error = 1;
	size_t sizeForOne = size / nRanks, start_index = size / nRanks * myRank;
        cudaMemcpyToSymbol(add, &addH, sizeof(double));
        cudaMallocHost(&A, sizeof(double) * size * size);
        cudaMallocHost(&Anew, sizeof(double) * size * size);
	fill(A,Anew,size);

	// выделяем необходимую для процесса память на GPU
	if(nRanks!=1){
		if (myRank != 0 && myRank != nRanks - 1) sizeForOne += 2;
		else sizeForOne += 1;
	 }
	cudaMalloc((void**)&d_A, size * sizeForOne * sizeof(double));
	cudaMalloc((void**)&d_Anew, size * sizeForOne * sizeof(double));
	cudaMalloc((void**)&d_Asub, size * sizeForOne * sizeof(double));

	size_t offset = (myRank != 0) ? size : 0;
 	cudaMemcpy(d_A, (A + (start_index * size) - offset), sizeof(double) * size * sizeForOne, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Anew, Anew + (start_index * size) - offset, sizeof(double) * size * sizeForOne, cudaMemcpyHostToDevice);

	// создаем потоки и назначаем приоритет
	int leastPriority = 0;
        int greatestPriority = leastPriority;
	cudaStream_t stream_boundaries, stream_inner;
	cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
	cudaStreamCreateWithPriority(&stream_boundaries, cudaStreamDefault, greatestPriority);
	cudaStreamCreateWithPriority(&stream_inner, cudaStreamDefault, leastPriority);

        dim3 threadPerBlock((size<1024)? size:1024, 1);
        dim3 blocksPerGrid(size / ((size<1024)? size:1024), sizeForOne);

	double* d_error;
        cudaMalloc(&d_error, sizeof(double));

	// определяем требования к временному хранилищу устройства и выделяем память
    	void     *d_temp_storage = NULL;
    	size_t   temp_storage_bytes = 0;
    	cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Asub, d_error, size*size);
    	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// цикл пересчета ошибки и обновления сетки 
	while((error > tol) && (iter < iter_max)){
		iter = iter + 1;
		
		// расчет границ
		update_boundaries<<<size, 1, 0, stream_boundaries>>>(d_A, d_Anew, size, sizeForOne);
		cudaStreamSynchronize(stream_boundaries);

		// расчет внутренних значений
		update<<<threadPerBlock, blocksPerGrid, 0, stream_inner>>>(d_A, d_Anew, size, sizeForOne);

		// пересчет значения ошибки раз в 150 итераций
		if (iter % 150 == 0){
			substract<<<threadPerBlock, blocksPerGrid, 0, stream_inner>>>(d_A, d_Anew, d_Asub, size, sizeForOne);
			cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Asub, d_error, size * sizeForOne, stream_inner);
			cudaStreamSynchronize(stream_inner);
			MPI_Allreduce((void*)d_error, (void*)d_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			cudaMemcpyAsync(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost, stream_inner);
		}
		
		// верхняя граница
        	if (myRank != 0){
		    MPI_Sendrecv(d_Anew + size + 1, size - 2, MPI_DOUBLE, myRank - 1, 0, 
					d_Anew + 1, size - 2, MPI_DOUBLE, myRank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// нижняя граница
		if (myRank != nRanks - 1){
		    MPI_Sendrecv(d_Anew + (sizeForOne - 2) * size + 1, size - 2, MPI_DOUBLE, myRank + 1, 0,
					d_Anew + (sizeForOne - 1) * size + 1, 
					size - 2, MPI_DOUBLE, myRank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		cudaStreamSynchronize(stream_inner);

		// обмен значениями
		double* swap = d_A;
		d_A = d_Anew;
		d_Anew = swap;
	}

    // освобождаем память
    cudaFree(d_A);
    cudaFree(d_Anew);
    cudaFree(d_error);
    cudaFree(d_Asub);
    cudaFree(d_error);
    cudaFree(d_temp_storage);
    MPI_Finalize();

    // считаем и выводим затраченное время с помощью std::chrono
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin1);
    if(myRank==0){
        std::cout << iter << ":" << error << "\n";
        std::cout << "The time:" << elapsed_ms.count() << "ms\n";
    }
	return 0;
}