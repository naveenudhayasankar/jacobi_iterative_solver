void generate_inputs(int n, double **A, double *b);

void jacobi(int rank, int proc, int n, double **A, double *b, double error, int max_iterations, int *itrs_used, double *x);
