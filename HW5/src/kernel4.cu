#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int *deviceData, float lowerX, float lowerY, float stepX, float stepY, int width, int count) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    
    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;
    float z_re = c_re, z_im = c_im;

    int i;
    if (count == 100000) {
        #pragma unroll
        for (i = 0; i < 100000; i++) 
        {
        
            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }else {
        for (i = 0; i < count; i++) 
        {
        
            if (z_re * z_re + z_im * z_im > 4.f)
                break;

            float new_re = z_re * z_re - z_im * z_im;
            float new_im = 2.f * z_re * z_im;
            z_re = c_re + new_re;
            z_im = c_im + new_im;
        }
    }


    deviceData[thisY * width + thisX] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    
    size_t imgSize = resX * resY;
    int *deviceData;
    cudaMalloc((void **)&deviceData, imgSize * sizeof(int));

    dim3 threadsPerBlock(8, 8);
    dim3 numBlock(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
   
    mandelKernel<<<numBlock, threadsPerBlock>>>(deviceData, lowerX, lowerY, stepX, stepY, resX, maxIterations);
    
    cudaMemcpy(img, deviceData, imgSize * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceData);
}
