/* 
 * Реализация уравнения теплопроводности в двумерной области
 * на равномерных сетках с использованием CUDA. 
 * Операция редукции (вычисление максимального значения ошибки)
 * реализуется с помощью библиотеки CUB.
*/

// подключение библиотек
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

using namespace std;
using namespace cub;

// функция, обновляющая значения сетки
__global__ void update(double* A, double* Anew, int size)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	if (j < size - 1 && j > 0 && i > 0 && i < size - 1){
		double left = A[i * size + j - 1];
		double right = A[i * size + j + 1];
		double top = A[(i - 1) * size + j];
		double bottom = A[(i + 1) * size + j];
		Anew[i*size + j] = 0.25 * (left + right + top + bottom);
	}
}

// функция нахождения разности двух массивов
__global__ void substract(double* A, double* Anew, double* res, int size){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if(i >= 0 && i < size && j >= 0 && j < size)
		res[i*size + j] = Anew[i*size + j] - A[i*size + j];
}

__constant__ double add;

// функция для заполнения массивов
__global__ void fill(double* A, double* Anew, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < size){
        A[i*(size) + 0] = 10 + add*i;
        A[i] = 10 + add*i;
        A[(size-1)*(size) + i] = 20 + add*i;
        A[i*(size)+size-1] = 20 + add*i;

        Anew[i*(size) + 0] = A[i*(size) + 0];
        Anew[i] = A[i];
        Anew[(size-1)*(size) + i] = A[(size-1)*(size) + i];
        Anew[i*(size)+size-1] = A[i*(size)+size-1];
    }
}

// основное тело программы
int main(int argc, char* argv[]){
    // время до выполнения программы
    auto begin = std::chrono::steady_clock::now();

    // выбираем устрйоство для исполнения
    cudaSetDevice(1);

    // инициализация переменных
    double tol = atof(argv[1]);
    const int size = atoi(argv[2]), iter_max = atoi(argv[3]);

    double *d_A = NULL, *d_Anew = NULL, *d_Asub;

    // выделение памяти под массивы и проверка на наличие ошибок
    cudaMalloc((void **)&d_A, sizeof(double)*size*size);
    cudaMalloc((void **)&d_Anew, sizeof(double)*size*size);
    cudaMalloc((void **)&d_Asub, sizeof(double)*size*size);

    // инициализация переменных
    int iter = 0;
    double error = 1;
    double addH = 10.0 / (size - 1);
    cudaMemcpyToSymbol(add, &addH, sizeof(double));

    dim3 threadPerBlock = dim3(32, 32);
    dim3 blocksPerGrid = dim3((size+threadPerBlock.x-1)/threadPerBlock.x,(size+threadPerBlock.y-1)/threadPerBlock.y);
    
    // заполняем сетки
    fill<<<blocksPerGrid, threadPerBlock>>>(d_A, d_Anew, size);

    // инициализация ошибки, редукции, потока и графа
    double* d_error;
    cudaMalloc(&d_error, sizeof(double));

    // определяем требования к временному хранилищу устройства
    void* d_temp_storage = NULL; // доступное для устройства выделение временного хранилища
    size_t temp_storage_bytes = 0; // размер выделяемой памяти для d_temp_storage
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_A, d_error, size*size); // предоставление количества байтов, необходимых для временного хранения, необходимого CUB
    // выделяем временное хранилище
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // цикл пересчета ошибки и обновления сетки 
    while((error > tol) && (iter < iter_max/2/100)) {
        iter = iter + 1;

        // если граф не создан, то создаем с помощью захвата цикла for
        if(!graphCreated)
	    {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            //обновление значений сетки (почему два раза - подробно описано в отчете)
            for(int i = 0; i<100;i++){
                update<<<blocksPerGrid, threadPerBlock,0,stream>>>(d_Anew,d_A, size);
                update<<<blocksPerGrid, threadPerBlock,0,stream>>>( d_A,  d_Anew,size);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated=true;
        }

        // запускаем граф
        cudaGraphLaunch(instance, stream);

        // вычитаем один массив из другого
        substract<<<blocksPerGrid, threadPerBlock,0,stream>>>(d_A, d_Anew, d_Asub, size);
        
        // находим новое значение ошибки
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Asub, d_error, size*size,stream);
        cudaMemcpy(&error, d_error, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << iter << ":" << error << "\n";

    }

    // освобождаем память
    cudaFree(d_A);
    cudaFree(d_Anew);
    cudaFree(d_error);

    // считаем и выводим затраченное время с помощью std::chrono
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin);
    std::cout << "The time:" << elapsed_ms.count() << "ms\n";
    return 0;
}