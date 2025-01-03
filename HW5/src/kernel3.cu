#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int *deviceData, float lowerX, float lowerY, float stepX, float stepY, int pitch, int group, int count) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = (blockIdx.x * blockDim.x + threadIdx.x) * group;
    int thisY = (blockIdx.y * blockDim.y + threadIdx.y) * group;
    
    for (int n = 0; n < group; n++){
        for(int m = 0; m < group; m++) {
            float c_re = lowerX + (thisX + m) * stepX;
            float c_im = lowerY + (thisY + n) * stepY;
            float z_re = c_re, z_im = c_im;

            int i;
            for (i = 0; i < count; i++) 
            {

                if (z_re * z_re + z_im * z_im > 4.f)
                    break;

                float new_re = z_re * z_re - z_im * z_im;
                float new_im = 2.f * z_re * z_im;
                z_re = c_re + new_re;
                z_im = c_im + new_im;
            }

            *((int *)((char *)deviceData + pitch * (thisY + n)) + (thisX + m)) = i;
        }
    }
   
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    size_t imgSize = resX * resY;
    int *hostData, *deviceData;
    cudaHostAlloc((void **)&hostData, imgSize * sizeof(int), cudaHostAllocMapped);
    // returns pitch
    size_t pitch = 0;
    cudaMallocPitch((void **)&deviceData, &pitch, resX * sizeof(int), resY);

    int group = 4;
    dim3 threadsPerBlock(4, 4);
    dim3 numBlock((resX / threadsPerBlock.x) / group, (resY / threadsPerBlock.y) / group);

    mandelKernel<<<numBlock, threadsPerBlock>>>(deviceData, lowerX, lowerY, stepX, stepY, pitch, group, maxIterations);

    cudaMemcpy2D(hostData, resX * sizeof(int), deviceData, pitch, resX * sizeof(int), resY, cudaMemcpyDeviceToHost);
    memcpy(img, hostData, imgSize * sizeof(int));

    cudaFree(deviceData);
    cudaFreeHost(hostData); 
}
