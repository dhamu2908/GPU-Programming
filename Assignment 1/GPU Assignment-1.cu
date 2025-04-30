
/**
 *   CS6023: GPU Programming
 *   Assignment 1
 *
 *   Please don't change any existing code in this file.
 *
 *   You can add your code whereever needed. Please add necessary memory APIs
 *   for your implementation. Use cudaFree() to free up memory as soon as you're
 *   done with an allocation. This will ensure that you don't run out of memory
 *   while running large test cases. Use the minimum required memory for your
 *   implementation. DO NOT change the kernel configuration parameters.
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <math.h>

using std::cin;
using std::cout;


__global__ void CalculateInvertedGrayScale(long int *d_red, long int *d_green, long int *d_blue, long int *d_T1, int m, int n){

    int id = blockIdx.x * blockDim.x + threadIdx.x ;

    if( id < m * n ){

      float temp = ( d_red [ id ] + d_green [ id ] + d_blue [ id ] ) / (3.0);

      int row =  ( m - 1 ) - ( id / n ) ;

      row = row * n ;

      int col = id % n ;

      int index  = row + col ;

      d_T1 [ index ] = floor ( temp );

    }

}

__global__ void CalculateThomasTransformation(long int *d_red, long int *d_green, long int *d_blue, long int *d_T2, int m, int n){

    int id = blockIdx.x * blockDim.x + threadIdx.x ;

    if( id < m * n){

    float green = d_green [ id ];

    float green1 = sqrt ( green );

    int  green_2 = floor ( green1 );

    double temp =  (0.5 * d_red [ id ] ) + green_2 + d_blue [ id ];

    d_T2 [ id ] = floor ( temp );

    }

}



int main(int argc, char **argv)
{
    const char *inputFileName = argv[1] ;

    FILE *inputFile = NULL;



    // Open the file for reading
    if ((inputFile = fopen(inputFileName, "r")) == NULL) {
        printf("Failed at opening the file %s\n", inputFileName);
        return 1; // Exit with an error code
    }
    int m,n;
    fscanf (inputFile, "%d %d", &m, &n) ;


    long int *red = new long int[m * n]; /* red channel */
    long int *green = new long int[m * n]; /* green channel */
    long int *blue = new long int[m * n]; /* blue channel */
    long int *T1 = new long int[m * n];
    long int *T2 = new long int[m * n];

    int num;
    int channel = 0;
    long int counter = 0;
    while (fscanf(inputFile, "%d", &num) != EOF) {
        switch (channel)
        {
            case 0:
            red[counter] = num;
            break;

            case 1:
            green[counter] = num;
            break;

            case 2:
            blue[counter] = num;
            break;
        }
        counter++;
        if(counter%(m*n) == 0)
        {
            channel++;
            counter = 0;
        }
    }




    /**
     *
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     *
     */

    long int *d_red;
    long int *d_green;
    long int *d_blue;
    long int *d_T1;
    long int *d_T2;

    // allocating memory
    cudaMalloc(&d_red, m * n * sizeof(long int));
    cudaMalloc(&d_green, m * n * sizeof(long int));
    cudaMalloc(&d_blue, m * n * sizeof(long int));
    cudaMalloc(&d_T1, m * n * sizeof(long int));
    cudaMalloc(&d_T2, m * n * sizeof(long int));

    // we need copy the matrices to the device matrices
    cudaMemcpy(d_red, red, m * n * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_green, green, m * n * sizeof(long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blue, blue, m * n * sizeof(long int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024, 1, 1);
    dim3 blocksPerGrid(ceil(m * n / 1024.0), 1, 1);

    auto start = std::chrono::high_resolution_clock::now();
    //write function
    //Note that d_T1 has the resultant matrix of Inverted Gray Scale matrix
    //Note that d_T2 has the resultant matrix of Thomas Transformation matrix
    CalculateInvertedGrayScale<<<blocksPerGrid, threadsPerBlock>>>(d_red, d_green, d_blue,d_T1,m,n);
    CalculateThomasTransformation<<<blocksPerGrid, threadsPerBlock>>>(d_red, d_green, d_blue,d_T2,m,n);
    cudaDeviceSynchronize();

    cudaMemcpy(T1, d_T1, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    cudaMemcpy(T2, d_T2, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
    cudaFree(d_T1);
    cudaFree(d_T2);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed1 = end - start;

    // Make sure your final output from the device is stored in d_T1.

    /**
     *
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     *
     */


    std::ofstream file("cuda.out");
    if (file.is_open())
    {
        for (long int i = 0; i < m; i++)
        {
            for (long int j = 0; j < n; j++)
            {
                file << T1[ i* n + j] << " ";
            }
            file << "\n";
        }
        for (long int i = 0; i < m; i++)
        {
            for (long int j = 0; j < n; j++)
            {
                file << T2[i * n + j] << " ";
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
