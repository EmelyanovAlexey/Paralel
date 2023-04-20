#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <stdlib.h>
#include <openacc.h>
#include <cublas_v2.h>

// края рамок
constexpr double ANGLE1 = 10;
constexpr double ANGLE2 = 20;
constexpr double ANGLE3 = 30;
constexpr double ANGLE4 = 20;
constexpr double negOne = -1;

int main(int argc, char *argv[])
{
    int n = 0;
    int MAX_ITERATION = 0;
    double ACCURACY = 0;
    double error = 1.0;
    int cntIteration = 0;
    int max_idx = 0; // индекс максимального элемента
    int cntUpdate = 0; // счетчик обновлений ошибки

    cublasStatus_t status; // переменная для хранения статуса выполнения функций из CUBLAS
    cublasHandle_t handle; // дескриптор CUBLAS
    cublasCreate(&handle); // инициализация дескриптора CUBLAS

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
    double *inter = new double[n * n];

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

// Копирование массивов arrNew, arr и inter на устройство для использования в расчетах
#pragma acc enter data copyin(arrNew[:n * n], arr[:n * n], inter[:n * n])
    while (cntIteration < MAX_ITERATION && error > ACCURACY)
    {
// Параллельный цикл для расчета новых значений массива arrNew на GPU с использованием arr и с асинхронным выполнением
#pragma acc parallel loop collapse(2) present(arrNew[:n * n], arr[:n * n]) vector_length(128) async
        for (size_t i = 1; i < n - 1; i++)
        {
            for (size_t j = 1; j < n - 1; j++)
            {
                // Расчет новых значений для каждой ячейки массива arrNew на основе соответствующих значений в arr
                arrNew[i * n + j] = 0.25 * (arr[(i + 1) * n + j] + arr[(i - 1) * n + j] + arr[i * n + j - 1] + arr[i * n + j + 1]);
            }
        }
        double *copy = arr;
        arr = arrNew;
        arrNew = copy;

        if (cntUpdate >= 200 && cntIteration < MAX_ITERATION)
        {

#pragma acc data present(inter[:n * n], arrNew[:n * n], arr[:n * n]) wait
// Запись результатов расчетов inter, arrNew и arr с устройства на хост для дополнительной обработки
            {
#pragma acc host_data use_device(arrNew, arr, inter)
// Использование устройства для выполнения CUBLAS-функций на CPU
                {
                    status = cublasDcopy(handle, n * n, arr, 1, inter, 1); // Копирование массива arr в inter с помощью CUBLAS-функции
                    status = cublasDaxpy(handle, n * n, &negOne, arrNew, 1, inter, 1); // Вычисление разности между arrNew и arr и сохранение результата в inter с помощью CUBLAS-функции
                    status = cublasIdamax(handle, n * n, inter, 1, &max_idx); // Нахождение индекса максимального элемента в inter с помощью CUBLAS-функции
                }
            }

// Запись максимального значения из inter на устройстве в переменную error
#pragma acc update self(inter[max_idx])
            error = fabs(inter[max_idx]);
            cntUpdate = 0;
        }
        cntIteration++;
        cntUpdate++;
    }

    std::cout << "iteration: " << cntIteration << " \n"
              << "error: " << error << "\n";
    delete[] arr;
    delete[] arrNew;
    return 0;
}
