#include <iostream>
#include <cmath>
#include <boost/math/special_functions/factorials.hpp>

// using Taylor series of arctan(1)
double pi_summation_slow(int N) 
{
  double sum = 0.0;
  int sign = 1;

  for (int i=0; i<N; ++i)
  {
    sum += sign/(2*i + 1.0);
    sign = -sign;
  }

  return 4*sum;
}

// Using Euler's formula
// $$
// \arctan(x) = \sum_{n=0}^\infty \frac{2^{2n} (n!)^2}{(2n + 1)!} \frac{x^{2n + 1}}{(1 + x^2)^{n + 1}}
// $$
// and Machin's formula
// $$
// \frac{\pi}{4} = 4 \arctan\frac{1}{5} - \arctan\frac{1}{239}
// $$
double pi_summation_fast_overflow(int order) {
  using boost::math::factorial;
  double sum = 0.0;

  for (unsigned int n=0; n<order; ++n) {
    double f = factorial<double>(n);
    double common = pow(2.0, 2*n)*f*f/factorial<double>(2*n + 1);
    double A = pow(25./26., n+1)/pow(5., 2*n+1);
    double B = pow(239.*239. / (239.*239. + 1.), n+1)/pow(239., 2*n+1);
    sum += common*( 4*A - B );
  }

  return 4*sum;
}

// Taylor series approximation of sin(x)
// Each term is calculated explicitly
double taylor_sin(double x, int order)
{
  using boost::math::factorial;
  double sum = 0.0;
  int sign = 1;

  for (unsigned int n=0; n<order; ++n)
  {
    sum += sign*pow(x, 2*n + 1)/factorial<double>(2*n +1);
    sign = -sign;
  }

  return sum;
}

// Optimised Taylor series approximation of sin(x)
// Each term is calculated from the previous one
double taylor_sin_opt(double x, int order)
{
  using boost::math::factorial;
  double sum = x;
  double an = x;

  for (unsigned int n=1; n<order; ++n)
  {
    an = -x*x*an/(2*n*(2*n+1));
    sum += an;
  }

  return sum;
}

int main()
{
  // pi_summation_slow(10);
  // pi_summation_slow(100);
  // pi_summation_slow(1000);
  // pi_summation_slow(10000);
  // pi_summation_slow(100000);
  // pi_summation_slow(1000000);
  // pi_summation_slow(10000000);

  // pi_summation_fast_overflow(1);
  // pi_summation_fast_overflow(2);
  // pi_summation_fast_overflow(3);
  // pi_summation_fast_overflow(4);
  // pi_summation_fast_overflow(5);
  // pi_summation_fast_overflow(6);

  taylor_sin(M_PI, 10);
  taylor_sin(2*M_PI, 10);
  // taylor_sin(2*M_PI, 20); // overflow error

  taylor_sin_opt(M_PI, 10);
  taylor_sin_opt(2*M_PI, 10);
  taylor_sin_opt(2*M_PI, 20);
  taylor_sin_opt(2*M_PI, 30);
  taylor_sin_opt(4*M_PI, 10);
  taylor_sin_opt(4*M_PI, 20);
  taylor_sin_opt(4*M_PI, 30);
  taylor_sin_opt(4*M_PI, 40);
}