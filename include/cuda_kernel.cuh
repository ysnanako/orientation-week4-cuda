#ifndef CUDA_KERNEL_CUH
#define CUDA_KERNEL_CUH

#include <bits/stdc++.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void CUDA_CD(float *, float *, float *, float *, int);
__global__ void CUDA_A(float *, float *, float *, int);
__global__ void CUDA_B(float *, float *, int);
void CUDA_init(vector<float> &, vector<float> &);

#endif // CUDA_KERNEL_CUH