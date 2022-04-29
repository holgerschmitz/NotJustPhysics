#include <iostream>
#include <math.h>

typedef double real;

template<int rank>
struct Array {
  int val[rank];
};

namespace device {
  template<typename T>
  __device__
  const T& min(const T& a, const T& b) {
    return (a>b) ? b : a;
  }
}

template<int rank>
struct GridStride {
  int innerDim[rank];
  int offset[rank];
  int outerDim[rank];

  __host__ __device__
  int getInnerCount() const {
    int result = innerDim[0];
    for (int i=1; i<rank; ++i) {
      result *= innerDim[i];
    }
    return result;
  }

  __host__ __device__
  void innerPosFromInnerCount(int count, int (&pos)[rank]) const {
    for (int i=rank-1; i>=0; --i) {
      pos[i] = count % innerDim[i];
      count = count / innerDim[i];
    }
  }

  __host__ __device__
  int outerCountFromInnerPos(int pos[rank]) const {
    int count = pos[0] + offset[0];
    for (int i=1; i<rank; ++i) {
      count = count*outerDim[i-1] + pos[i] + offset[i];
    } 
    return count;
  }
};

template<int rank>
struct LocalGridIterator {
  int nIter;
  int delta;
  int start;
  int end;
  int i;
  int j;
  int skip0;
  int t0;

  __device__
  LocalGridIterator(const GridStride<rank> &stride) {
    nIter = stride.getInnerCount();
    delta = (nIter - 1) / (blockDim.x * gridDim.x) + 1;
    start = delta*(blockIdx.x * blockDim.x + threadIdx.x);
    end = device::min(start + delta, nIter);

    int innerPos[2];
    stride.innerPosFromInnerCount(start, innerPos);

    skip0 = stride.outerDim[0] - stride.innerDim[0];
    t0 = stride.offset[1] + stride.innerDim[1]
      + stride.outerDim[1]*(stride.offset[0] + innerPos[0]);


    j = stride.outerCountFromInnerPos(innerPos);
    i = start;
  }
};

// Kernel function to add the elements of two arrays
__global__
void add(GridStride<2> stride, real dx, real *x, real *y)
{ 
  LocalGridIterator<2> iter(stride);

  int innerPos[2];
  stride.innerPosFromInnerCount(iter.start, innerPos);

  while (iter.i < iter.end) {
    
    y[iter.j] = (x[iter.j+1] + x[iter.j-1] + x[iter.j+stride.outerDim[0]] + x[iter.j-stride.outerDim[0]] - 4*x[iter.j]) / (dx*dx); 
    
    ++iter.i;
    if (++iter.j >= iter.t0) {
      iter.j += iter.skip0;
      iter.t0 += stride.outerDim[0];
    }
  }
}

template<typename T>
T testFunc(T x, T y) {
  real r2 = x*x + y*y;
  return exp(-r2);
//    return 10000*x + y;
}

int main(void)
{
  int D = 200;
  int N = D*D;
  real *x, *y;
  real dx = real(0.5f/D);

  GridStride<2> stride{{D - 8, D - 8}, {4, 4}, {D, D}};

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(real));
  cudaMallocManaged(&y, N*sizeof(real));

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
  real maxError = -1.0f;
  real comparedTo;
  int maxI, maxJ;

  for (int i = 0; i < D; ++i) {
    for (int j = 0; j < D; ++j) {
//      real r2 = (i*i + j*j)*dx*dx;
//      real expected = (i<4 || j<4 || i>=D-4 || j>=D-4) ? 0.0f : 4.0f*(r2 - 1)*exp(-r2);
      real expected = (i<4 || j<4 || i>=D-4 || j>=D-4) ? 0.0f : (
        testFunc<real>(i*dx - dx, j*dx) + testFunc<real>(i*dx + dx, j*dx) + 
        testFunc<real>(i*dx, j*dx - dx) + testFunc<real>(i*dx, j*dx + dx) 
        - 4.0*testFunc(i*dx, j*dx)
      ) /(dx*dx);
    
      real err = fabs(y[i*D + j] - expected);
      if (err > maxError) {
        maxError = err;
        comparedTo = expected;
        maxI = i;
        maxJ = j;
      }
      std::cout << i*dx << " " << j*dx << " " << y[i*D + j] << std::endl;
    }
    std::cout << std::endl;
  }

  std::cerr << "Max error: " << maxError << " " << maxI << " " << maxJ << " " << y[maxI*D + maxJ] << " " << comparedTo << std::endl;

//   real xf = maxI*dx;
//   real yf = maxJ*dx;

//   real expectedAtMax = // testFunc(xf - dx, yf); 
// ( testFunc(xf - dx, yf) + testFunc(xf + dx, yf) + 
//   testFunc(xf, yf - dx) + testFunc(xf, yf + dx)
//   - 4.0*testFunc(xf, yf)
// )/(dx*dx);

//   std::cout << "Expected value: " << expectedAtMax << std::endl;

//   // double check, get it? ;)
//   double xd = maxI*dx;
//   double yd = maxJ*dx;
  
//   double precise = // testFunc(xd - dx, yd); 
// ( testFunc(xd - dx, yd) + testFunc(xd + dx, yd) + 
//   testFunc(xd, yd - dx) + testFunc(xd, yd + dx) 
//   - 4.0*testFunc(xd, yd)
// )/(double(dx)*double(dx));

//   std::cout << "Better value: " << precise << std::endl;
  // Free memory
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
