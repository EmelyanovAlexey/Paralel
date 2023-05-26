/* -------------------------------------------------------------*
* Простая нейронная сеть, реализованная с помощью CUDA и cuBLAS *
* --------------------------------------------------------------*/

// подключение библиотек
#include <cmath>
#include <string>
#include <cstdio>
#include <vector>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define WITH_ACTIVATION true
#define WITHOUT_ACTIVATION false
#define printResult true
#define DONT_PRINT false
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// макросы для проверки ошибок CUDA и CUBLAS 
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// макросы для проверки ошибок CUBLAS 
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)


// функция активации.
__global__ void nn_Sigmoid(float *arr, int size)
{
	int id = threadIdx.x;
	if(id < size - 1 && id > 0) 
		arr[id] = 1 / (1 + exp(-arr[id]));
}

// функция проверяет, совпадает ли заданное значение с предопределенным шаблоном.
void check(float result){
	float pattern = round(0.5696*1000)/1000;
	result = round(result*1000)/1000;
	// Сравниваем шаблон и округленное значение
	if(pattern == result) std::cout << "IT'S RIGHT ANSWER!"<< std::endl;
	// Выводим сообщение о том, что результат не совпадает с шаблоном
	else std::cout << "result("<< result << ") != pattern(" << pattern << ")" << std::endl;
}

// класс для линейного слоя и сигмоиды
class NN
{
private:
	cublasHandle_t handle;
	float alpha, beta;
	float *weights, *biases, *output;
	int inputSize, outputSize;
	bool isActivation;

	// считывание весов из файла
	void readWeights(std::string pathToWeights){
		float *host_array = new float [inputSize*outputSize], *host_array_row = new float [inputSize*outputSize];
		try{
			std::ifstream fin(pathToWeights);
			for (int i = 0; i < inputSize*outputSize; i++) fin >> host_array_row[i];
			fin.close();
		}
		catch(std::exception const& e){
			std::cout << "error: " << e.what() << std::endl;
		}
		for(int i=0;i<inputSize;i++){
			for(int j=0;j<outputSize;j++){
				host_array[i*outputSize+j] = host_array_row[IDX2C(i,j, inputSize)];
			}
		}
		CUDA_CHECK(cudaMalloc(&weights, outputSize * inputSize * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(weights, host_array, outputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));
		delete[] host_array, host_array_row;
	};

	// считывание добавочных членов из файла
	void readBiases(std::string pathToWeights){
		float *host_array = new float [outputSize];
		try{
			std::ifstream fin(pathToWeights);
			for (int i = 0; i < outputSize; i++) fin >> host_array[i];
			fin.close();
		}
		catch(std::exception const& e){
			std::cout << "There was an error: " << e.what() << std::endl;
		}
		CUDA_CHECK(cudaMalloc(&biases, outputSize * sizeof(float)));
		CUDA_CHECK(cudaMemcpy(biases, host_array, outputSize*sizeof(float), cudaMemcpyHostToDevice));
		delete[] host_array;
	};

public:
	// конструкторы
	NN(){
		inputSize = 0;
		outputSize = 0;
		alpha = 1.0;
		beta = 1.0;
		isActivation = false;
	};

	NN(std::string pathToWeights, std::string pathToBiases, int inSize, int outSize, bool activation){
		alpha = 1.0;
		beta = 1.0;
		inputSize = inSize;
		outputSize = outSize;
		readWeights(pathToWeights);
		readBiases(pathToBiases);
		isActivation = activation;
	};

	// сигмоида
	void Sigmoid(float *arr)
	{
		nn_Sigmoid<<<1, outputSize>>> (arr, outputSize);
	};

	// реализация линейного слоя
	float* Linear(float* input){
		// Создаем экземпляр объекта cublasHandle_t для использования библиотеки cuBLAS
		CUBLAS_CHECK(cublasCreate(&handle));
		// Выполняем матрично-векторное умножение с использованием функции cublasSgemv из библиотеки cuBLAS.
		// CUBLAS_OP_N - указывает, что матрица весов 'weights' и вектор входных данных 'input' не нужно транспонировать перед умножением.
		// outputSize - количество строк в матрице весов, inputSize - количество столбцов в матрице весов,
		// &alpha - масштабный множитель, weights - матрица весов, outputSize - количество столбцов в матрице весов,
		// input - вектор входных данных, 1 - шаг между элементами вектора входных данных,
		// &beta - масштабный множитель для вектора смещений (biases), biases - вектор смещений, 1 - шаг между элементами вектора смещений.
		CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_N, outputSize, inputSize, &alpha, 
             weights,outputSize, input, 1, &beta, biases, 1));
		// Уничтожаем объект cublasHandle_t после использования
		CUBLAS_CHECK(cublasDestroy(handle));
		// Если флаг isActivation истинный, то применяем функцию активации Sigmoid к вектору смещений (biases).
		if(isActivation){
			Sigmoid(biases);
		}
		return biases;
	};

	// деструктор
	~NN(){
		if(weights!=nullptr) cudaFree(weights);
		if(biases!=nullptr) cudaFree(biases);
	};
};

// класс для построения сети из функций NN. По умолчанию поставлена сеть из ТЗ,
// но есть возможность сделать свою.
class Net
{
private:
	float *array;
	int inputSize, outputSize;
	std::vector<NN> layers;

	// чтение входного массива из файла
	void readInput(std::string pathToWeights){
		float *host_array = new float [inputSize];
		// Открываем файл по указанному пути pathToWeights для чтения
		try{
			std::ifstream fin(pathToWeights);
			// Читаем значения из файла и сохраняем их в массив
			for (int i = 0; i < inputSize; i++) fin >> host_array[i];
			fin.close();
		}
		// В случае возникновения исключения выводим сообщение об ошибке
		catch(std::exception const& e){
			std::cout << "There was an error: " << e.what() << std::endl;
		}

		// Выделяем память на устройстве CUDA с использованием функции cudaMalloc.
		CUDA_CHECK(cudaMalloc(&array, inputSize * sizeof(float)));
		// Копируем данные из массива host_array на устройство CUDA с использованием функции cudaMemcpy.
		// cudaMemcpyHostToDevice указывает на направление копирования данных из хоста на устройство.
		CUDA_CHECK(cudaMemcpy(array, host_array, inputSize*sizeof(float), cudaMemcpyHostToDevice));
		delete[] host_array;
	};

	// печать
	void printResult(float* arr){
		float* host_array = new float[outputSize];
		CUDA_CHECK(cudaMemcpy(host_array, arr, outputSize*sizeof(float), cudaMemcpyDeviceToHost));
		std::cout << "Result: " << std::endl;
		for (int i = 0; i < outputSize; i++)
		{
			std::cout << host_array[i] << std::endl;
		}
		check(host_array[0]);
		delete[] host_array;
	};

public:
	// конструктор по умолчанию
	Net(){
		inputSize = 1024;
		outputSize = 1;
	};

	// // добавление слоя в сеть
	// void pushBackLinear(std::string pathToWeights, std::string pathToBiases, int inSize, int outSize, bool activation){
	// 	if(layers.size()==0) inputSize=inSize;
	// 	outputSize = outSize;
	// 	layers.push_back(NN(pathToWeights, pathToBiases, inSize, outSize, activation));
	// };

	// // запуск сети
	// void myForward(std::string pathToFile, bool print){
	// 	readInput(pathToFile);
	// 	for(auto& layer : this->layers){array = layer.Linear(array);}
	// 	if(print) printResult(array);
	// };

	// запуск базовой сети
	void forward(std::string pathToFile, bool print){
		readInput(pathToFile);
		NN layer1("./data/weights1.bin", "./data/biases1.bin", 1024, 256, WITH_ACTIVATION);
		array = layer1.Linear(array);
		NN layer2("./data/weights2.bin", "./data/biases2.bin", 256, 16, WITH_ACTIVATION);
		array = layer2.Linear(array);
		NN layer3("./data/weights3.bin", "./data/biases3.bin", 16, 1, WITH_ACTIVATION);
		array = layer3.Linear(array);
		if(print) printResult(array);
	}

	// деструктор
	~Net(){
		if(array!=nullptr) cudaFree(array);
	};
};

int main()
{
	Net model;
	model.forward("./data/inputs1.bin", printResult);
	return 0;
}