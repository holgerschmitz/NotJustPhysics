#include <iostream>
#include <math.h>

// Kernel function to add the elements of two arrays
__global__
void add(int a0, int a1, int s0, int s1, int d0, int d1, float dx, float *x, float *y)
{ 
  int delta = a0*a1 / (blockDim.x * gridDim.x) + 1;
  int start = delta*(blockIdx.x * blockDim.x + threadIdx.x);
  int end = start + delta;
  if (end >= a0*a1) {
    end =  a0*a1 - 1;
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
      j += skip0 - 1;
    }
  }
}

int main(void)
{
  int D = 1<<10;
  int N = D*D;
  float *x, *y;
  float dx = float(4.0f/D);

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < D; ++j) {
      float r2 = (i*i + j*j)*dx*dx;
      x[i*D + j] = exp(-r2);
    }
  }

  for (int i = 0; i < N; ++i) {
    y[i] = 0.0f;
  }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(D - 8, D - 8, 4, 4, D, D, dx, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = -1.0f;
  int maxI, maxJ;

  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < D; ++j) {
      float r2 = (i*i + j*j)*dx*dx;
      float expected = (i<4 || j<4 || i>=D-4 || j>=D-4) ? 0.0f : 4.0f*(r2 - 1)*exp(-r2);
      
      float err = fabs(y[i*D + j] - expected);
      if (err > maxError) {
        maxError = err;
        maxI = i;
        maxJ = j;
      }
//      std::cout << i*dx << " " << j*dx << " " << y[i*D + j] << std::endl;
    }
  }

  std::cout << "Max error: " << maxError << " " << maxI << " " << maxJ << " " << y[maxI*D + maxJ] << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
