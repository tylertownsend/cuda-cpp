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

#include "utils.h"

const int BLOCK_SIZE = 1024

__global__
void shared_reduce_kernel_min(const float* const d_logLuminance,
                              float* d_write_reduce_to,
                              const int input_size) {
  extern __shared__ float shared_output[]

  int global_id = threadIdx.x + blockDim.x * blockIdx.x;
  int local_id = threadIdx.x;

  // thread_copies_input_data_to shared_memory
  if (global_id >= input_size) {
    shared_output[local_id] = d_write_reduce_to[0];
  } else {
    shared_output[local_id] = d_write_reduce_to[global_id];
  }
  __synchthreads();
  
  
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_id < s) {
      shared_output[local_id] = min(shared_output[local_id],
                                    shared_output[local_id + s]);
    }
    __synchthreads();
  }

  if (local_id == 0) {
    d_write_reduce_to[blockIdx.x] = shared_data[0];
  }
}

__global__
void shared_reduce_kernel_max(const float* const d_logLuminance,
                              float* d_write_reduce_to,
                              const int input_size) {
  extern __shared__ float shared_output[]

  int global_id = threadIdx.x + blockDim.x * blockIdx.x;
  int local_id = threadIdx.x;

  // thread_copies_input_data_to shared_memory
  if (global_id >= input_size) {
    shared_output[local_id] = d_write_reduce_to[0];
  } else {
    shared_output[local_id] = d_write_reduce_to[global_id];
  }
  __synchthreads();
  
  
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (local_id < s) {
      shared_output[local_id] = max(shared_output[local_id],
                                    shared_output[local_id + s]);
    }
    __synchthreads();
  }

  if (local_id == 0) {
    d_write_reduce_to[blockIdx.x] = shared_data[0];
  }
}

__global__
void histogram_calculation(unsigned int* out_histo, 
                           const float* d_, 
                           int num_bins, 
                           int input_size,
                           float min_val,
                           float range_vals) {
  int thread_id = threadIdx.x;
  int global_id = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (global_id >= input_size) {
    return;
  }

  int bin = (int)((d_[global_id] - min_val) / (float)range_vals) * num_bins;
  atomicAdd(&(out_histro[bin]), 1);
}

void find_max_and_min(const float* const d_logLuminance,
                      float& min_logLum,
                      float& max_logLum,
                      int input_size) {
  int threads_per_block = BLOCK_SIZE;
  int output_size= (input_size - 1)/BLOCK_SIZE + 1;
  dim3 block_size = dim3(threads_per_block, 1, 1);
  dim3 grid_size = dim3(output_size, 1, 1);

  float* d_reduce_input;
  checkCudaErrors(cudaMalloc((void**)&d_reduce_input,
                             input_size * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_reduce_input,
                             d_logLuminance,
                             input_size * sizeof(float),
                             cudaMemcpyDeviceToDevice));
  
  while (output_size > 1) {
    float* d_reduce_output;
    checkCudaErrors(cudaMalloc((void**)&d_reduce_output),
                               output_size * sizeof(float));
    shared_reduce_kernel_min<<<grid_size, block_size>>>(d_reduce_output,
                                                        d_reduce_intput,
                                                        input_size);
    checkCudaErrors(cudaFree(d_reduce_input));

    input_size = output_size;
    output_size = (input_size - 1) / BLOCK_SIZE + 1;
    block_size = dim3(threads_per_block, 1, 1);
    grid_size = dim3(output_size); 
    d_reduce_input = d_reduce_output;
  }

  float* h_reduce_output= (float *)malloc(sizeof(float));
  checkCudaErrors(cudaMemcpy(h_reduce_output, d_reduce_ouput, sizeof(float), cudaMemcpyDeviceToHost));
  min_logLum = h_reduce_output[0];
}
                      

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  find_max_and_min(d_logLuminance, min_logLum, max_logLum);
}
