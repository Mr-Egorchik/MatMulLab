#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>

//a - mxp, b - pxn, c - mxn

void matmul_cpu(const double* a, const double* b, double* c, const int m, const int n, const int p) {
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < p; ++k)
                c[i * n + j] += a[i * p + k] * b[k * n + j];
}

__global__ void matmul_gpu(const double* a, const double* b, double* c, const int m, const int n, const int p) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < m && col < n) {
        double temp = 0;
        for (size_t k = 0; k < p; ++k)
            temp += a[row * p + k] * b[k * n + col];
        c[row * n + col] = temp;
    }
}

bool assert_mm(const double* ha, const double* da, int m, int n) {
    for (size_t i = 0; i < m * n; ++i)
        if (ha[i] != da[i]) {
            std::cout << "Assert false";
            return false;
        }
    std::cout << "Assert true";
    return true;
}

int main()
{
    srand(time(0));

    int m = 2000;
    int n = 2000;
    int p = 2000;

    double* a = new double[m * p];
    double* b = new double[p * n];
    double* c = new double[m * n];

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < p; ++j)
            a[i * p + j] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

    for (int i = 0; i < p; ++i)
        for (int j = 0; j < n; ++j)
            b[i * n + j] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            c[i * n + j] = 0;

    clock_t start, end;

    start = clock();
    matmul_cpu(a, b, c, m, n, p);
    end = clock();

    double cpu_time = static_cast <double>(end - start) / static_cast <double>(CLOCKS_PER_SEC);

    double* hc = new double[m * n];

    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            hc[i * n + j] = 0;

    double* da, *db, *dc;
    cudaMalloc(&da, m * p * sizeof(double));
    cudaMalloc(&db, p * n * sizeof(double));
    cudaMalloc(&dc, m * n * sizeof(double));

    cudaMemcpy(da, a, m * p * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, p * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, m * n * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block_dim(32, 32);
    dim3 grid_dim(ceil(static_cast <double> (n) / static_cast <double> (block_dim.x)), ceil(static_cast <double> (m) / static_cast <double> (block_dim.y)));

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    cudaEventRecord(begin, 0);
    matmul_gpu << <grid_dim, block_dim >> > (da, db, dc, m, n, p);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, begin, stop);

    cudaMemcpy(hc, dc, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    assert_mm(c, hc, m, n);

    std::cout << "\nCPU time(s):\t" << cpu_time;
    std::cout << "\nGPU time(s):\t" << gpu_time/1000.;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    cudaEventDestroy(begin);
    cudaEventDestroy(stop);

    delete[] a;
    delete[] b;
    delete[] c;
    delete[] hc;

    return 0;
}