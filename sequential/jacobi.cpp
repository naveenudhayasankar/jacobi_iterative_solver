#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<chrono>
#include<iomanip>
#include "jacobi.hpp"

using namespace std;

int main(int argc, char *argv[]){
    int n, i;
    n = atoi(argv[1]);

    auto start = chrono::high_resolution_clock::now();
    double *b; 
    b  = new double [n];
    double **A;
    A = new double* [n];
    for(i = 0; i < n; i++){
        A[i] = new double [n];
    }
    generate_inputs(n, A, b);
    double *x;
    x = new double [n];
    for(i = 0; i < n; i++){
        x[i] = 0.0;
    }
    double error = 1.0e-2;
    int max_iterations = 30000;
    int count = 0;
    cout << "********** Solving a system of linear equations with " << n << " unknowns **********" << endl;
    jacobi(n, A, b, error, max_iterations, &count, x);
    cout << "********** Solution computed in " << count << " iterations **********" << endl;
    double sum = 0.0;
    for(i = 0; i < n; i++){
        double diff = x[i] - 1.0;
        sum += (diff >= 0.0) ? diff : -diff;
    }
    cout << "********** Error is " << sum << " **********" << endl;
    auto end = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    time_taken *= 1e-9;
    cout << "********** Runtime is " << time_taken << setprecision(9) << " seconds **********" << endl;
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

void jacobi(int n, double **A, double *b, double error, int max_iterations, int *itrs_used, double *x){
    double *dx, *y;
    dx = new double [n];
    y = new double [n];
    int i, j, k;
    for(k = 0; k <= max_iterations; k++){
        double sum = 0.0;
        for(i = 0; i < n; i++){
            dx[i] = b[i];
            for(j = 0; j < n; j++){
                dx[i] -= A[i][j] * x[j];
            }
            dx[i] /= A[i][i];
            y[i] += dx[i];
            sum += (dx[i] >= 0.0) ? dx[i] : -dx[i];
        }
        for(i = 0; i < n; i++){
            x[i] = y[i];
        }
        if(sum < error){
            break;
        }
    }
    *itrs_used = k + 1;
    delete dx;
    delete y;
}