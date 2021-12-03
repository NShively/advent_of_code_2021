#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


/**
 * CUDA Kernel Device Code
 * Just checks the value in the array before it and increments a counter if it 
 */
__global__ void 
vectorGreaterThan(const int *A, bool* B, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i <= 0 || i >= size) {
        return;
    }
    if (A[i] > A[i-1]) {
        B[i-1] = true;
    }
    
}

__global__ void
vector3dSum(const int *A, int* B, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i <= 1 || i >= size) {
        return;
    }
    B[i-2] = A[i] + A[i-1] + A[i-2];
//    printf("test val b: i[%d] B[%d]\n", i, B[i-2]); 
}

int main() {
    std::vector<int> inData;
    std::ifstream File;
    File.open("input.txt");
    while(!File.eof())
    {
        int p = 0;
        File >> p;
        if (File.eof()) break;
        inData.push_back(p);
    }
    int numElements = inData.size();
    size_t sizeInput = numElements * sizeof(int);
    size_t sizeOutput = (numElements-1) * sizeof(bool);
    size_t sizeOutput2 = 2*(numElements-2) * sizeof(int);
    int *host_input = (int*)malloc(sizeInput);
    bool *host_output = (bool*)malloc(sizeOutput);
    host_input = inData.data();   
    cudaError_t err = cudaSuccess;
    int * d_input = NULL;
    err = cudaMalloc((void**) &d_input, sizeInput);
    if (err != cudaSuccess) {
        std::cout<< "Errormallocing device input!" << std::endl;
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    bool * d_output = NULL;
    err = cudaMalloc((void**) &d_output, sizeOutput);
    if (err != cudaSuccess) {
        std::cout<< "Errormallocing device input!" << std::endl;
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    err = cudaMemcpy(d_input, host_input, sizeInput, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cout << "Failed to copy from host to device: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    int threadsPerBlock=256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorGreaterThan<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, sizeInput);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cout << "vectorGreaterThan kernel error: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(host_output, d_output, sizeOutput, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cout << "Failed to copy result back to host: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    int total = 0;
    for (int i = 0; i < sizeOutput; i++) {
        if (host_output[i]) total++;
    }
    std::cout << "part A total: " << total << std::endl;

    int* host_output_pt2 = (int*)malloc(sizeOutput2);
    int* d_output_2 = NULL;
    err = cudaMalloc((void**)&d_output_2, sizeOutput2);
    if (err != cudaSuccess)
    {
        std::cout << "error pt2 malloc " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    vector3dSum<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output_2, numElements);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cout << "error in pt2 kernl: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(host_output_pt2, d_output_2, sizeOutput2, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cout << "error copy back to host pt2: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    int total2 = 0;
    for (int i = 1; i < numElements-2; i++)
    {
        if (host_output_pt2[i] > host_output_pt2[i-1]) total2++;
    }
    std::cout << "part B total: " << total2 << std::endl;
}
