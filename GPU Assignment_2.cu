#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>
//#include <cuda/cuda_runtime.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;

__global__ void dkernel(long int *matrix, long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    // sample kernel you can use your own kernel

    // Shared Memory allocation
    extern __shared__ long int shared_memory[];
    
    int filter_size = r * s * c; 
    long int *shared_filter = shared_memory;
    
    int res_x = blockIdx.x * blockDim.x + threadIdx.x;
    int res_y = blockIdx.y * blockDim.y + threadIdx.y;
    int filter_id = blockIdx.z;

    // Load filter into shared memory
    for (int i = threadIdx.x; i < filter_size; i += blockDim.x) {
        shared_filter[i] = filter[filter_id * filter_size + i];
    }

    __syncthreads(); 

    int shared_mem_available = (49152 / sizeof(long int)) - filter_size; // Remaining shared memory

    long int *shared_image = shared_memory + filter_size;

    // Storing the entires channels in shared memory which can fits completely
    for (int ch = 0; ch < shared_mem_available / (h * w); ch++) {
        for (int idx = threadIdx.x; idx < h * w; idx += blockDim.x) {
            shared_image[ch * h * w + idx] = matrix[(ch * h + (idx / w)) * w + (idx % w)];
        }
    }

    __syncthreads(); 

    // Performing convolution 
    if (res_x < w && res_y < h) {
        long int sum = 0;
        for (int ch = 0; ch < c; ch++) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < s; j++) {

                    int in_x = j + res_x - s / 2 ;
                    int in_y = i + res_y - r / 2 ;

                    long int sum1 = 0;

                    if (in_x >= 0 && in_x < w && in_y >= 0 && in_y < h) {
                        if (ch < shared_mem_available / (h * w)) {
                            sum1 = shared_image[(ch * h + in_y) * w + in_x]; // From shared memory
                        } else {
                            sum1 = matrix[in_x + (in_y + ch * h) * w]; // From global memory
                        }
                    }

                    long int sum2 = shared_filter[(ch * r + i) * s + j];

                    sum += sum1 * sum2;
                }
            }
        }
        result[(filter_id * h + res_y) * w + res_x] = sum;
    }
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++)
    {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++)
    {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     **/

    auto start = std::chrono::high_resolution_clock::now(); // keep it just before the kernel launch

    /****************************************************Start Here***********************************************************/

    /**
        Do device allocations, kernel launches and copying everything here
        and the final answer should be stored back in h_ans, use cudaFree to free up the allocated memory on GPU
    */

    dim3 threads_per_block(32, 32); 
    dim3 blocks_per_grid(ceil(w / 32.0), ceil(h / 32.0), k); 

    // Shared memory size
    size_t shared_memory_size = 49152; 

    
    long int *d_mat, *d_filter, *d_ans;
    
    cudaMalloc(&d_mat, h * w * c * sizeof(long int));
    cudaMalloc(&d_filter, r * s * c * k * sizeof(long int));
    cudaMalloc(&d_ans, h * w * k * sizeof(long int));

    
    cudaMemcpy(d_mat, h_mat, h * w * c * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, r * s * c * k * sizeof(long int), cudaMemcpyHostToDevice);

    // Launching the kernel
    dkernel<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(d_mat, d_filter, d_ans, h, w, c, r, s, k);

    // Copy result back to host
    cudaMemcpy(h_ans, d_ans, h * w * k * sizeof(long int), cudaMemcpyDeviceToHost);

    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    auto end = std::chrono::high_resolution_clock::now(); // keep it just after the kernel launch
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */

    cudaDeviceSynchronize();
    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < h * k; i++)
        {
            for (long int j = 0; j < w; j++)
            {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open())
    {
        file2 << elapsed1.count() << "\n";
        file2.close();
    }
    else
    {
        std::cout << "Unable to open file";
    }

    return 0;
}
