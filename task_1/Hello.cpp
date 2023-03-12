#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>

const unsigned int N = 10000000;

float arrFloat[N];
float sumFloat = 0;

// double arrDouble[N];
// double sumDouble = 0;

#pragma acc enter data create(arrFloat[0:N], sumFloat)
// #pragma acc enter data create(arrDouble[0:N], sumDouble)

// float
// #pragma acc parallel loop present(arrFloat[0:N],sumFloat)
void ctreateArrFloat(float *arr)
{
    float pi = 3.1415926535;
    float x;
    x = 2 * pi / (N - 1);
    for (unsigned int i = 0; i < N; i++)
        arr[i] = sin(x * i);
}

// сумма
void sumArrFloat(float *arr)
{
    sumFloat = 0;
    #pragma acc parallel loop present(arrFloat[0:N],sumFloat) reduction(+:sumFloat)
    for (unsigned int i = 0; i < N; i++) sumFloat += arr[i];
    #pragma acc exit data delete (arrFloat[0:N],sumFloat) reduction(+:sumFloat)
}

// void printArrFloat(float *arr)
// {
//     for (unsigned int i = 0; i < N; i++)
//         std::cout << arr[i] << " ";
// }

// bouble
// #pragma acc parallel loop present(arrDouble[0:N],sumDouble)
// void ctreateArrDouble(double *arr)
// {
//     double pi = 3.1415926535;
//     double x;
//     x = 2 * pi / (N - 1);
//     for (unsigned int i = 0; i < N; i++)
//         arr[i] = sin(x * i);
// }

// // сумма
// void sumArrFloat(double *arr)
// {
//     sumDouble = 0;
//     #pragma acc parallel loop present(arrDouble[0:N],sumDouble) reduction(+:sumDouble)
//     for (unsigned int i = 0; i < N; i++) sumDouble += arr[i];
//     #pragma acc exit data delete (arrDouble[0:N],sumFloat) reduction(+:sumFloat)
// }

// void printArrDouble(double *arr)
// {
//     for (unsigned int i = 0; i < N; i++)
//         std::cout << arr[i] << " ";
// }

int main()
{
    // float
    ctreateArrFloat(arrFloat);
    // printArrFloat(arrFloat);

    // bouble
    // ctreateArrDouble(arrDouble);
    // printArrDouble(arrDouble)

    return 0;
}

// Заполнить на
// графическом процессоре массив типа float / double значениями синуса(один период на всю
//     длину массива).Размер массива - 10 ^ 7 элементов.Для заполненного массива на графическом процессоре посчитать сумму всех элементов массива.
//     Сравнить со значением, вычисленном на центральном процессоре.Сравнить разультат для
//     массивов  float и double.
//
//     При сборке программы для исполнения на GPU использовать компилятор pgcc / pgc++ с ключами
//     `-acc - Minfo = accel`. Необходимо
//     разобраться с выводом о распараллеливании кода.
//
//     Произвести профилирование программы установив переменную окружения PGI_ACC_TIME = 1. Необходимо
//     понять сколько времени тратится на вычисление и сколько времени на передачу
//     данных.
//
//     Результаты работы, какие знания о работе программы на GPU были получены представить в виде отчета.
//
//     Возможные дополнительные вопросы :
//
//*как выполняется параллельное суммирование,
//
//*почему в программе два цикла, а ядер(по выводу профилировщика) три
//
//* почему сумма всех элементов не равна нулю

// Директива parallel позволяет пометить участок кода, содержащий параллелизм. Встретив эту директиву, компилятор создаст ядро для
// исполнения на графическом процессоре.

// Директивой kernels указывается, что код может содержать параллелизм, и компилятор