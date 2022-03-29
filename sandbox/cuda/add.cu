#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int n, int a0, int a1, int s0, int s1, int d0, int d1, float dx, float *x, float *y)
{ 
  int delta = n / (blockDim.x * gridDim.x) + 1;
  int start = delta*(blockIdx.x * blockDim.x + threadIdx.x);
  int end = start + delta;
  if (end >= n) {
    end = n - 1;
  }
  
  int p0 = start % a0 + s0;
  int q0 = start / a0;
  int p1 = q0 % a1 + s1;

  int skip0 = d0 - s0 - a0;
  int t0 = s0 + a0;

  int j = p0 + d0*p1;

  int i = start;
  while (i < end) {
    
    y[j] = (x[j+1] + x[j-1] + x[j+d0] + x[j-d0] - 4*x[j]) / (dx*dx);
    
    ++i;
    if (++j >= t0) {
      j += skip0;
    }
  }
}

int main(void)
{
  int N = 1<<26;
  float *x, *y;
  float dx = float(2*M_PI/N);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = sin(i*dx);
    y[i] = 0.0f;
  }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, dx, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;

  for (int i = 0; i < N-1; i++) {
    maxError = fmax(maxError, fabs(y[i]-cos((i + 0.5f)*dx)));
//    std::cout << i*dx << " " << y[i] << " " << maxError << std::endl;
  }


  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
