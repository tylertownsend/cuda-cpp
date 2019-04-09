/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/
#include <cmath>
#include <cstdlib>
#include <unordered_map>
#include <string>

#include <cuda_runtime.h>

#include "utils.h"

const int BLOCK_SIZE = 1024;
const std::string MAX = "MAX";
const std::string MIN = "MIN";
typedef float (*reduce_callback)(float, float);

__device__
float max_val(float value_1, float value_2) {
  return value_1 < value_2 ? value_2 : value_1;
}

__device__
float min_val(float value_1, float value_2) {
  return value_1 < value_2 ? value_1 : value_2;
}

// __device__ 
// callback_map get_reduce_function{
//   {"MAX", max_val},
//   {"MIN", min_val}
// };

void print_device_array(const float* const d_array, int size) {
  float* h_array = (float*)malloc(size * sizeof(float));
  checkCudaErrors(cudaMemcpy(h_array, d_array, size*sizeof(float),
                             cudaMemcpyDeviceToHost));
  printf("Array \n");
  for (int i = 0; i < size; ++i) {
    printf("%.3f%c", h_array[i], ((i == size - 1) ? '\n' : ' '));
  }
  free(h_array);
}

__global__
void reduce_kernel(const float* const d_input_array,
                          float* d_output_array,
                          const int input_size,
                          bool find_max) {
  extern __shared__ float shared_output[];

  int global_id = threadIdx.x + blockDim.x * blockIdx.x;
  int local_id = threadIdx.x;

  if (global_id >= input_size) {
    shared_output[local_id] = d_input_array[0];
  } else {
    shared_output[local_id] = d_input_array[global_id];
  }
  __syncthreads(); 
  
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_id < s) {
      if (find_max) {
        shared_output[local_id] = max_val(shared_output[local_id],
                                          shared_output[local_id + s]);
      } else {
        shared_output[local_id] = min_val(shared_output[local_id],
                                          shared_output[local_id + s]);
      }
    }
    __syncthreads();
  }

  if (local_id == 0) {
    d_output_array[blockIdx.x] = shared_output[0];
  }
}

__global__
void histogram_calculation(unsigned int* d_histo, 
                           const float* d_logLuminance, 
                           int num_bins, 
                           int input_size,
                           float min_val,
                           float range_vals) {
  int global_id = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (global_id >= input_size) {
    return;
  }

  int bin = (int)((d_histo[global_id] - min_val) / (float)range_vals) * num_bins;
  atomicAdd(&(d_histo[bin]), 1);
}

__global__
void blelloch_scan_single_block(unsigned int* d_cdf, int num_bins) {
  extern __shared__ unsigned int scan_array[];
  int thread_id = threadIdx.x;
  scan_array[thread_id] = d_cdf[thread_id];
  
  int stride = 1;
  while (stride <= num_bins) {
    int index = (threadIdx.x + 1)*stride*2 - 1;
    if (index <  2*BLOCK_SIZE) {
      scan_array[index] += scan_array[index - stride];
    }
    stride = stride * 2;
    __syncthreads();
  }

  if (thread_id == 0) {
    scan_array[num_bins - 1] = 0;
  }

  stride = BLOCK_SIZE / 2;
  while (stride > 0) {
    int index = (thread_id + 1) * stride*2 - 1;
    if((index + stride) < 2 * BLOCK_SIZE) {
      scan_array[index + stride] += scan_array[index];
    }
    stride = stride/2;
    __syncthreads();
  }

  d_cdf[thread_id] = scan_array[thread_id];
}

unsigned int* compute_histogram(const float* const d_logLuminance,
                                int num_bins,
                                int input_size,
                                int min_logLum,
                                int range) {
  unsigned int* d_histo;
  checkCudaErrors(cudaMalloc(&d_histo, num_bins * sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_histo, 0, num_bins * sizeof(unsigned int)));

  int threads_per_block = BLOCK_SIZE;
  int output_size = 1 + (input_size - 1) / BLOCK_SIZE;
  dim3 block_size = dim3(threads_per_block, 1, 1);
  dim3 grid_size = dim3(output_size);
  
  histogram_calculation<<<grid_size, block_size>>>(d_histo,
                                                   d_logLuminance,
                                                   num_bins,
                                                   input_size,
                                                   min_logLum,
                                                   range);

  unsigned int* h_histo = (unsigned int*) malloc(num_bins * sizeof(unsigned int));
  checkCudaErrors(cudaMemcpy(h_histo, d_histo, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_histo));
  return h_histo;
}

void find_max_and_min(const float* const d_logLuminance,
                      float& val_logLum,
                      int input_size,
                      bool find_max) {
  int threads_per_block = BLOCK_SIZE;
  int output_size = 1 + (input_size - 1) / BLOCK_SIZE;

  float* d_input;
  checkCudaErrors(cudaMalloc((void**)&d_input,
                             input_size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_input,
                             d_logLuminance,
                             input_size * sizeof(float),
                             cudaMemcpyDeviceToDevice));
  float* d_output;
  while (input_size > 1) {
    checkCudaErrors(cudaMalloc((void**)&d_output,
                               output_size * sizeof(float)));

    reduce_kernel<<<output_size, threads_per_block, threads_per_block * sizeof(float)>>>(
      d_input, d_output, input_size, find_max);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaFree(d_input));
    input_size = output_size;
    output_size = (input_size - 1) / BLOCK_SIZE + 1;
    d_input = d_output;
  }
  
  float* h_output= (float *)malloc(sizeof(float));
  checkCudaErrors(cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaFree(d_output));
  val_logLum = h_output[0];
}
                      

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  // This particular image had 240 cols and 294 rows
  float* h_logLuminance = (float*)malloc(numRows*numCols*sizeof(float));
  checkCudaErrors(cudaMemcpy(h_logLuminance,d_logLuminance,numRows*numCols*sizeof(float),
                             cudaMemcpyDeviceToHost));
  printf("\n\nREFERENCE CALCULATION (Min/Max): \n");
  float logLumMin = h_logLuminance[0];
  float logLumMax = h_logLuminance[0];
  printf("num rows %d\n", numRows);
  printf("num cols %d\n", numCols);
  for (size_t i = 1; i < numCols * numRows; ++i) {
    logLumMin = min(h_logLuminance[i], logLumMin);
    logLumMax = max(h_logLuminance[i], logLumMax);
  }
  printf("  Min logLum: %f\n  Max logLum: %f\n",logLumMin,logLumMax);
  free(h_logLuminance);

  int input_size = numRows*numCols;
  find_max_and_min(d_logLuminance, min_logLum, input_size, false);
  find_max_and_min(d_logLuminance, max_logLum, input_size, true);

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  printf("%f\n", min_logLum);
  printf("%f\n", max_logLum);


  int range = max_logLum - min_logLum;

  unsigned int* h_histo = compute_histogram(d_logLuminance, numBins, 
                                            input_size, min_logLum, range);
  
  checkCudaErrors(cudaMemcpy(d_cdf, h_histo, numBins*sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

  blelloch_scan_single_block<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(d_cdf, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
