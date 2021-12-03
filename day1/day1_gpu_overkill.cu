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


int main() {
    std::vector<int> inData;
    std::ifstream File;
    File.open("input.txt");
    while(!File.eof())
    {
        // Now you may be saying 'Nick, why arent you just incrementing a counter every time a value increases right here, rather than do extra work?
        // Because fuck you, lets use this gpu for something it has no business doing
        int p = 0;
        File >> p;
        if (File.eof()) break;
        inData.push_back(p);
        std::cout << p << std::endl;
    }
    int numElements = inData.size();
    size_t sizeInput = numElements * sizeof(int);
    size_t sizeOutput = (numElements-1) * sizeof(bool);
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
    std::cout << "total: " << total << std::endl;

}
