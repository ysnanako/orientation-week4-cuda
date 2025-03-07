#include "cuda_kernel.cuh"
using namespace std;

#define length 1000

void CUDA_init(vector<float> &a, vector<float> &b)
{
    /// *** Initialize thread_num, blocks_num *** ///
    unsigned int vector_size = a.size();
    size_t threads_per_block = 512;
    size_t number_of_blocks = (vector_size + threads_per_block - 1) / threads_per_block;
    vector<float> c(vector_size);
    vector<float> d(vector_size);

    /// *** Create CUDA variables *** ///
    float *d_vector_a, *d_vector_b, *d_vector_c, *d_vector_d;
    cudaMalloc((void **)&d_vector_a, vector_size * sizeof(float));
    cudaMalloc((void **)&d_vector_b, vector_size * sizeof(float));
    cudaMalloc((void **)&d_vector_c, vector_size * sizeof(float));
    cudaMalloc((void **)&d_vector_d, vector_size * sizeof(float));

    /// *** CD *** ///
    cudaMemcpy(d_vector_a, a.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, b.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_CD<<<number_of_blocks, threads_per_block>>>(d_vector_a, d_vector_b, d_vector_c, d_vector_d, vector_size);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), d_vector_c, vector_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d.data(), d_vector_d, vector_size * sizeof(float), cudaMemcpyDeviceToHost);

    /// *** A *** ///
    cudaMemcpy(d_vector_c, c.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_d, d.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_A<<<number_of_blocks, threads_per_block>>>(d_vector_a, d_vector_c, d_vector_d, vector_size);
    cudaDeviceSynchronize();
    cudaMemcpy(a.data(), d_vector_a, vector_size * sizeof(float), cudaMemcpyDeviceToHost);

    /// *** B *** ///
    cudaMemcpy(d_vector_a, a.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, b.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
    CUDA_B<<<number_of_blocks, threads_per_block>>>(d_vector_a, d_vector_b, vector_size);
    cudaDeviceSynchronize();
    cudaMemcpy(b.data(), d_vector_b, vector_size * sizeof(float), cudaMemcpyDeviceToHost);

    /// *** Free Space of CUDA variables  *** ///
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_vector_c);
    cudaFree(d_vector_d);
}
__global__ void CUDA_CD(float *d_a, float *d_b, float *d_c, float *d_d, int N)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= N)
        return;
    d_c[index] = d_a[index] + d_b[index];
    d_d[index] = d_a[index] - d_b[index];
}
__global__ void CUDA_A(float *d_a, float *d_c, float *d_d, int N)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= N)
        return;
    int row = index / length;
    int column = index % length;
    for(int i = 0; i < length; ++i)
        d_a[index] += d_c[row * length + i] * d_d[i * length + column];
}
__global__ void CUDA_B(float *d_a, float *d_b, int N)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= N)
        return;
    d_b[index] = d_a[index] + d_b[index];
}