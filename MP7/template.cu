// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

//@@ insert code here
__global__ void floattounsignedchar(float *input, unsigned char *ucharImage, int size){
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    if (ii < size){
        ucharImage[ii] = (unsigned char)(255 * input[ii]);
    }
}

__global__ void rgbtograyscale(unsigned char *input, unsigned char *grayImage, int size){
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    if (ii < size){
        unsigned char r = input[3 * ii];
        unsigned char g = input[3 * ii + 1];
        unsigned char b = input[3 * ii + 2];
        grayImage[ii] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

__global__ void histogramofgrayimage(unsigned char *input, unsigned int *output, int size){
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ unsigned int histo[HISTOGRAM_LENGTH];
    if (threadIdx.x < HISTOGRAM_LENGTH){
        histo[threadIdx.x] = 0;
    }
    __syncthreads();
    if (ii < size){
        atomicAdd(&(histo[input[ii]]), 1); // subtotal in each block
    }
    __syncthreads();
    if (threadIdx.x < HISTOGRAM_LENGTH){
        atomicAdd(&(output[threadIdx.x]), histo[threadIdx.x]); // add the subtotal to the global memory
    }
}

__global__ void cdfofhistogram(unsigned int *input, float *output, int size){
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float cdf[HISTOGRAM_LENGTH];
    if (ii < HISTOGRAM_LENGTH){
        cdf[ii] = input[ii];
    }
    __syncthreads();
    for (int stride = 1; stride < HISTOGRAM_LENGTH; stride *= 2){
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < HISTOGRAM_LENGTH){
            cdf[index] += cdf[index - stride];
        }
    }
    for (int stride = HISTOGRAM_LENGTH / 4; stride > 0; stride /= 2){
        __syncthreads();
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index + stride < HISTOGRAM_LENGTH){
            cdf[index + stride] += cdf[index];
        }
    }
    __syncthreads();
    if (ii < HISTOGRAM_LENGTH){
        output[ii] = (float)cdf[ii] * 1.0 / size;
    }
}

__global__ void histogramequalization(unsigned char *image, float *cdf, int size){
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    if (ii < size){
        float x = 255.0 * (cdf[image[ii]] - cdf[0]) / (1.0 - cdf[0]);
        float correct_color = min(max(x, 0.0), 255.0);
        image[ii] = (unsigned char)correct_color;
    }
}

__global__ void unsignedchartofloat(unsigned char *input, float *output, int size){
    int ii = threadIdx.x + blockIdx.x * blockDim.x;
    if (ii < size){
        output[ii] = (float)input[ii]/ 255;
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceInputImage;
  unsigned char *deviceChar;
  unsigned char *deviceGrayscale;
  unsigned int *deviceHistogram;
  float *deviceCDF;

  args = wbArg_read(argc, argv); 
  inputImageFile = wbArg_getInputFile(args, 0);
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //@@ insert code here
  wbTime_start(GPU, "Allocating GPU memory.");
  int imageSize = imageWidth * imageHeight * imageChannels;
  int graySize = imageWidth * imageHeight;
  cudaMalloc((void **)&deviceInputImage, imageSize * sizeof(float));
  cudaMalloc((void **)&deviceChar, imageSize * sizeof(unsigned char));
  cudaMalloc((void **)&deviceGrayscale, graySize * sizeof(unsigned char));
  cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **)&deviceCDF, HISTOGRAM_LENGTH * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceInputImage, hostInputImageData, imageSize * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  dim3 dimGrid((imageSize + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE), 1, 1);
  dim3 dimBlock2(BLOCK_SIZE * BLOCK_SIZE, 1, 1);
  dim3 dimGrid2((graySize + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE), 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  floattounsignedchar<<<dimGrid, dimBlock>>>(deviceInputImage, deviceChar, imageSize);
  cudaDeviceSynchronize();

  rgbtograyscale<<<dimGrid2, dimBlock2>>>(deviceChar, deviceGrayscale, graySize);
  cudaDeviceSynchronize();

  histogramofgrayimage<<<dimGrid2, dimBlock2>>>(deviceGrayscale, deviceHistogram, graySize);
  cudaDeviceSynchronize();

  cdfofhistogram<<<1, HISTOGRAM_LENGTH>>>(deviceHistogram, deviceCDF, graySize);
  cudaDeviceSynchronize();

  histogramequalization<<<dimGrid, dimBlock>>>(deviceChar, deviceCDF, imageSize);
  cudaDeviceSynchronize();

  unsignedchartofloat<<<dimGrid, dimBlock>>>(deviceChar, deviceInputImage, imageSize);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU.");
  cudaMemcpy(hostOutputImageData, deviceInputImage, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU.");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInputImage);
  cudaFree(deviceChar);
  cudaFree(deviceGrayscale);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, outputImage);
  //@@ insert code here
  return 0;
}
