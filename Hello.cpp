#include <iostream>

const unsigned int N = 10000000;

float arrFloat[N];
//double arrDouble[N];

// float
#pragma acc kernels
void ctreateArrFloat(float* arr) {
    float pi = 3.1415926535;
    float x;

    x = 2 * pi / (N - 1);
    for (unsigned int i = 0; i < N; i++) {
        arr[i] = sin(x * i);
    }
}

void printArrFloat(float* arr) {
    for (unsigned int i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
    }
}

// bouble
#pragma acc kernels
void ctreateArrDouble(double* arr) {
    double pi = 3.1415926535;
    double x;

    x = 2 * pi / (N - 1);
    for (unsigned int i = 0; i < N; i++) {
        arr[i] = sin(x*i);
    }
}

void printArrDouble(double* arr) {
    for (unsigned int i = 0; i < N; i++) {
        std::cout << arr[i] << " ";
    }
}

int main()
{
    // float
    ctreateArrFloat(arrFloat);
    printArrFloat(arrFloat);

    // bouble
    //ctreateArrDouble(arrDouble);
    //printArrDouble(arrDouble)

    return 0;
}


//Заполнить на
//графическом процессоре массив типа float / double значениями синуса(один период на всю
//    длину массива).Размер массива - 10 ^ 7 элементов.Для заполненного массива на графическом процессоре посчитать сумму всех элементов массива.
//    Сравнить со значением, вычисленном на центральном процессоре.Сравнить разультат для
//    массивов  float и double.
//
//    При сборке программы для исполнения на GPU использовать компилятор pgcc / pgc++ с ключами
//    `-acc - Minfo = accel`. Необходимо
//    разобраться с выводом о распараллеливании кода.
//
//    Произвести профилирование программы установив переменную окружения PGI_ACC_TIME = 1. Необходимо
//    понять сколько времени тратится на вычисление и сколько времени на передачу
//    данных.
//
//    Результаты работы, какие знания о работе программы на GPU были получены представить в виде отчета.
//
//    Возможные дополнительные вопросы :
//
//*как выполняется параллельное суммирование,
//
//*почему в программе два цикла, а ядер(по выводу профилировщика) три
//
//* почему сумма всех элементов не равна нулю
