#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void add(int* a, int* b, int* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    int a[] = {1, 2, 3};
    int b[] = {4, 5, 6};
    int c[3] = {0}; // or use sizeof(a)/sizeof(int) if more general

    int *cudaA, *cudaB, *cudaC;

    // Allocate memory on GPU
    cudaMalloc((void**)&cudaA, sizeof(a));
    cudaMalloc((void**)&cudaB, sizeof(b));
    cudaMalloc((void**)&cudaC, sizeof(c));

    // Copy data from host to device
    cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and 3 threads
    add<<<1, 3>>>(cudaA, cudaB, cudaC);

    // Copy result back to host
    cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < 3; i++) {
        printf("%d ", c[i]);
    }
    printf("\n");

    // Free GPU memory
    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);

    return 0;
}
