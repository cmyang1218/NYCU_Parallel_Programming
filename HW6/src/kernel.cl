__kernel void convolution(__global const float *inputImage, __global const float *filter, __global float *outputImage, 
                            const int imageHeight, const int imageWidth, const int filterWidth) 
{
    int idx = get_global_id(0);
    int row = idx / imageWidth;
    int col = idx % imageWidth;
    int halffilterSize = filterWidth / 2;

    float sum = 0.0f;
    
    for (int i = -halffilterSize; i <= halffilterSize; i++) {   
        for (int j = -halffilterSize; j <= halffilterSize; j++) {
            if (row + i >= 0 && row + i < imageHeight 
                && col + j >= 0 && col + j < imageWidth) {
                sum += (inputImage[(row + i) * imageWidth + (col + j)] 
                        * filter[((i + halffilterSize) * filterWidth + (j + halffilterSize))]);
            }
        }
    }
    outputImage[row * imageWidth + col] = sum;
}
