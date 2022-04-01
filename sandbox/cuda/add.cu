#include <iostream>
#include <math.h>

template<int rank>
struct GridStride {
  int a[rank];
  int s[rank];
  int d[rank];
};

// Kernel function to add the elements of two arrays
__global__
void add(GridStride<2> stride, float dx, float *x, float *y)
{ 
  int nIter = stride.a[0]*stride.a[1];
  int delta = nIter / (blockDim.x * gridDim.x) + 1;
  int start = delta*(blockIdx.x * blockDim.x + threadIdx.x);
  int end = start + delta;
  if (end >= nIter) {
    end =  nIter;
  }
  
  int p0 = start % stride.a[0] + stride.s[0];
  int q0 = start / stride.a[0];
  int p1 = q0 % stride.a[1] + stride.s[1];

  int skip0 = stride.d[0] - stride.s[0] - stride.a[0];
  int t0 = stride.s[0] + stride.a[0];

  int j = p0 + stride.d[0]*p1;

  int i = start;
  while (i < end) {
    
    y[j] = (x[j+1] + x[j-1] + x[j+stride.d[0]] + x[j-stride.d[0]] - 4*x[j]) / (dx*dx); // + x[j+d0] + x[j-d0] - 4*x[j]) / (dx*dx);
    
    ++i;
    if (++j >= t0) {
      j += skip0 - 1;
    }
  }
}

template<typename T>
T testFunc(T x, T y) {
  float r2 = x*x + y*y;
  return exp(-r2);
//    return 10000*x + y;
}

int main(void)
{
  int D = 1<<10;
  int N = D*D;
  float *x, *y;
  float dx = float(4.0f/D);

  GridStride<2> stride{{D - 8, D - 8}, {4, 4}, {D, D}};

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < D; ++j) {
      x[i*D + j] = testFunc(i*dx, j*dx);
    }
  }

  for (int i = 0; i < N; ++i) {
    y[i] = 0.0f;
  }

  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(stride, dx, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors
  float maxError = -1.0f;
  float comparedTo;
  int maxI, maxJ;

  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < D; ++j) {
//      float r2 = (i*i + j*j)*dx*dx;
//      float expected = (i<4 || j<4 || i>=D-4 || j>=D-4) ? 0.0f : 4.0f*(r2 - 1)*exp(-r2);
      float expected = (i<4 || j<4 || i>=D-4 || j>=D-4) ? 0.0f : (
        testFunc<float>(i*dx - dx, j*dx) + testFunc<float>(i*dx + dx, j*dx) + 
        testFunc<float>(i*dx, j*dx - dx) + testFunc<float>(i*dx, j*dx + dx) 
        - 4.0*testFunc(i*dx, j*dx)
      ) /(dx*dx);
    
      float err = fabs(y[i*D + j] - expected);
      if (err > maxError) {
        maxError = err;
        comparedTo = expected;
        maxI = i;
        maxJ = j;
      }
//      std::cout << i*dx << " " << j*dx << " " << y[i*D + j] << std::endl;
    }
  }

  std::cout << "Max error: " << maxError << " " << maxI << " " << maxJ << " " << y[maxI*D + maxJ] << " " << comparedTo << std::endl;

  float xf = maxI*dx;
  float yf = maxJ*dx;

  float expectedAtMax = // testFunc(xf - dx, yf); 
( testFunc(xf - dx, yf) + testFunc(xf + dx, yf) + 
  testFunc(xf, yf - dx) + testFunc(xf, yf + dx)
  - 4.0*testFunc(xf, yf)
)/(dx*dx);

  std::cout << "Expected value: " << expectedAtMax << std::endl;

  // double check, get it? ;)
  double xd = maxI*dx;
  double yd = maxJ*dx;
  
  double precise = // testFunc(xd - dx, yd); 
( testFunc(xd - dx, yd) + testFunc(xd + dx, yd) + 
  testFunc(xd, yd - dx) + testFunc(xd, yd + dx) 
  - 4.0*testFunc(xd, yd)
)/(double(dx)*double(dx));

  std::cout << "Better value: " << precise << std::endl;
  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
