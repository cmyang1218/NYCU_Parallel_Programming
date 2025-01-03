#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth;
    int imageSize = imageHeight * imageWidth;
   
    cl_command_queue commandQueue = clCreateCommandQueue(*context, *device, 0, NULL);
    // create memory buffer on device.
    cl_mem inputImageMem = clCreateBuffer(*context, CL_MEM_READ_ONLY, imageSize * sizeof(float), NULL, NULL);
    cl_mem filterMem = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize * sizeof(float), NULL, NULL);
    cl_mem outputImageMem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, imageSize * sizeof(float), NULL, NULL);
    
    // copy inputImage and filter to buffer.
    cl_int ret;
    ret = clEnqueueWriteBuffer(commandQueue, inputImageMem, CL_TRUE, 0, 
            imageSize * sizeof(float), inputImage, 0, NULL, NULL);
    CHECK(ret, "enqueue Image Buffer");
    ret = clEnqueueWriteBuffer(commandQueue, filterMem, CL_TRUE, 0, 
            filterSize * sizeof(float), filter, 0, NULL, NULL);
    CHECK(ret, "enqueue Filter Buffer");

    cl_kernel kernelFunction = clCreateKernel(*program, "convolution", NULL);
    clSetKernelArg(kernelFunction, 0, sizeof(cl_mem), (void *)&inputImageMem);
    clSetKernelArg(kernelFunction, 1, sizeof(cl_mem), (void *)&filterMem);
    clSetKernelArg(kernelFunction, 2, sizeof(cl_mem), (void *)&outputImageMem);
    clSetKernelArg(kernelFunction, 3, sizeof(cl_int), (void *)&imageHeight);
    clSetKernelArg(kernelFunction, 4, sizeof(cl_int), (void *)&imageWidth);
    clSetKernelArg(kernelFunction, 5, sizeof(cl_int), (void *)&filterWidth);

    size_t globalItemSize = imageSize;
    size_t localItemSize = 128;
    status = clEnqueueNDRangeKernel(commandQueue, kernelFunction, 1, NULL, 
                                &globalItemSize, &localItemSize, 0, NULL, NULL);
    CHECK(status, "enqueue NDRange kernel");
    ret = clEnqueueReadBuffer(commandQueue, outputImageMem, CL_TRUE, 0,
            imageSize * sizeof(float), outputImage, 0, NULL, NULL);
    CHECK(ret, "enqueue Output Buffer");
}
