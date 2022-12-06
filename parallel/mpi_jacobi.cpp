#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include "mpi_jacobi.hpp"

using namespace std;

int main(int argc, char *argv[]){

    int n, i, numProc, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProc);

    double start = MPI_Wtime();
    n = atoi(argv[1]);

    if(rank == 0){
        cout << "********** Solving a system of linear equations with " << n << " unknowns **********" << endl;
    }
    
    double *b;
    b = new double [n];
    double **A;
    A = new double* [n];
    for(i = 0; i < n; i++){
        A[i] = new double [n];
    }
    generate_inputs(n, A, b);
    double *x;
    x = new double [n];
    for(i = 0; i < n; i++){
        x[i] = b[i]/2.0;
    }
    double error = 5e-2;
    int max_iterations = 5e+7;
    int count = 0;
    
    jacobi(rank, numProc, n, A, b, error, max_iterations, &count, x);
    if(rank == 0){
        cout << "********** Solution computed in " << count << " iterations **********" << endl;
        double sum = 0.0;
        for(i = 0; i < n; i++){
            double diff = x[i] - 1.0;
            sum += (diff >= 0.0) ? diff : -diff;
        }
        cout << "********** Error is " << sum << " **********" << endl;
        double end = MPI_Wtime();
        cout << "********** Runtime is " << end - start << " seconds **********" << endl;
    }
    MPI_Finalize();
    return 0;
}

void generate_inputs(int n, double **A, double *b){
    int i,j;
    for(i = 0; i < n; i++){
        b[i] = 2.0*n;
        for(j = 0; j < n; j++){
            A[i][j] = 1.0;
        }
        A[i][i] = n + 1.0;
    }
}

void jacobi(int rank, int proc, int n, double **A, double *b, double error, int max_iterations, int *itrs_used, double *x){
    double *dx, *y;
    dx = new double [n];
    y = new double [n];
    int i, j, k;
    double sum[proc];
    double total;
    int data_per_proc = n / proc;
    int start_ind = rank * data_per_proc;
    int end_ind = start_ind + data_per_proc;

    for(k = 0; k <= max_iterations; k++){
        sum[rank] = 0.0;
        for(i = start_ind; i < end_ind; i++){
            dx[i] = b[i];
            for(j = 0; j < n; j ++){
                dx[i] -= A[i][j] * x[j];
            }
            dx[i] /= A[i][i];
            y[i] += dx[i];
            sum[rank] += (dx[i] >= 0.0) ? dx[i] : -dx[i];
        }
        for(i = start_ind; i < end_ind; i++){
                x[i] = y[i];
            }
        double *temp_x;
        temp_x = new double[n];
        MPI_Allgather(MPI_IN_PLACE, data_per_proc, MPI_DOUBLE, x, data_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_DOUBLE, sum, 1, MPI_DOUBLE, MPI_COMM_WORLD); 
        total = 0.0;
        for(i = 0; i < proc; i++){
            total += sum[i];
        }
        if(rank == 0){
           // cout << "********** Error after " << k << " iterations: " << total << " **********" << endl;
        }
        if(total <= error){
            break;
        }
    }
    *itrs_used = k + 1;
    delete dx;
}
