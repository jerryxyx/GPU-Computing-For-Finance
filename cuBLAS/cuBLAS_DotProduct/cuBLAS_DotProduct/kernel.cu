
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <iostream>


const int n = 6;


int main()
{
	cublasHandle_t handle;
	
	float *x = new float[n];
	float *y = new float[n];


	for (int i = 0; i < n; i++)
	{
		x[i] = float(i);
		y[i] = float(i);
	}


	float *d_x, *d_y;

	cudaMalloc((void**)&d_x, n*sizeof(float));
	cudaMalloc((void**)&d_y, n*sizeof(float));

	cublasCreate(&handle);

	cublasSetVector(n, sizeof(float), x, 2, d_x, 2);
	cublasSetVector(n, sizeof(float), y, 1, d_y, 1);

	float result;

	cublasSdot(handle, n, d_x, 1, d_y, 1, &result);
	cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
	
	std::cout << "The Dot Product of x and y is :" << result << std::endl;
	for (int i = 0; i < n; i++) {
		std::cout << *(x+i) << std::endl;
	}
	

	cudaFree(d_x); cudaFree(d_y);
	cublasDestroy(handle);
	delete[] x; delete[] y;
	char temp;
	std::cin >> temp;
	return 0;

    return 0;
}
