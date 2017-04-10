#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
using namespace std;
#define INDEX2F(i,j,ld) (i+j*ld)
const int m = 2;
const int n = 3;

float main()
{
	cublasHandle_t handle;
	float *a, *x, *y;
	float *d_a, *d_x, *d_y;
	a = new float[m*n];
	x = new float[n];
	y = new float[m];
	cudaMalloc(&d_a, m*n * sizeof(float));
	cudaMalloc(&d_x, n * sizeof(float));
	cudaMalloc(&d_y, m * sizeof(float));
	int ind = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			a[i + j*m] = ind++;
		}
	}
	for (int i = 0; i < m; i++) {
		y[i] = 0;
	}
	for (int i = 0; i < n; i++) {
		x[i] = 1;
	}
	cout << "A matrix is:" << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << a[i + j*m]<<" ";
		}
		cout << endl;
	}

	cublasCreate(&handle);
	cublasSetMatrix(m, n, sizeof(float), a, m, d_a, m);
	cublasSetVector(n, sizeof(float), x, 1, d_x, 1);
	cublasSetVector(m, sizeof(float), y, 1, d_y, 1);
	float alpha, beta;
	alpha = 1.0f;
	beta = 1.0f;
	cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_a, m, d_x, 1,&beta, d_y, 1);
	cublasGetVector(m, sizeof(float), d_y, 1, y, 1);
	
	for (int i = 0; i < m; i++) {
		cout << y[i] << endl;
	}

	cudaFree(d_a);
	cudaFree(d_x);
	cudaFree(d_y);
	delete[] x;
	delete[] y;
	delete[] a;
	cublasDestroy(handle);

	char temp;
	cin >> temp;


    return 0;
}

