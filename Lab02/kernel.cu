
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace std;




__global__ void neighboredPairSum(int* input, int n, int step) {
    int tid = threadIdx.x; // Thread ID in the block
    int offset = 1 << step; // Calculate the offset for each step
    if (tid % (2 * offset) == 0 && (tid + offset) < n) {
        input[tid] += input[tid + offset];
    }
    __syncthreads(); // Thread synchronization
}



void displayList(int* ptr, int size) {
    cout << "{ ";

    //print list
    for (auto i = 0; i < size; i++)
    {
        cout << ptr[i] << ", ";
    }

    cout << "}";
}

int main()
{
    //init variables
    int N = 10;
    int  *input_host, *input_gpu;
    int n_host, ste_host;

    //alocate space in host
    input_host = (int*)malloc(N * sizeof(int));


    //allocate space in GPU
    cudaMalloc(&input_gpu, N * sizeof(int));

    //attribute concecutive values to input host
    for (int i = 0; i < N; i++) {
        input_host[i] = i;
    }

    //print the initial list
    cout << "The initial list is: " << endl;
    displayList(input_host, N);


    //transfer data from host to gpu
    cudaMemcpy(input_gpu, input_host, N * sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel execution
    int threads_per_block = 128;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;

    // Perform neighbored pair sum reduction
    for (int j = 0; j < N; j++) {
        neighboredPairSum <<< num_blocks, threads_per_block >>> (input_gpu, N, j);
    }

    // Copy result back to host
    cudaMemcpy(input_host, input_gpu, N * sizeof(int), cudaMemcpyDeviceToHost);

    //print the result list
    cout << "The result list is: " << endl;
    displayList(input_host, N);

    // Clean up
    cudaFree(input_gpu);
    free(input_host);


    return 0;
}



