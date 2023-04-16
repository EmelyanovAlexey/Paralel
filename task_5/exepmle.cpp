#include <iostream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <stdlib.h>

#include <mpi.h>


// int main(int args, char* argv[]) {
//     int rank, size;
//     MPI_Init(&args, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     std::cout << "i am " << rank << " from " << size << "\n";
//     MPI_Finalize();
//     return 0;
// }



int main(int args, char* argv[]) {
    int rank, size;
    MPI_Init(&args, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int ack;

    if (rank == 0) {
        std::cout << "i am process" << "\n" << rank;
        ack = 1;
        MPI_Send(&ack, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else {
        MSI_Status status;
        MPI_Recv(&ack, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std::cout << "i am " << rank << " from " << size << "\n";
    }

    MPI_Finalize();
    return 0;
}