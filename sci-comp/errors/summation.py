#!/usr/bin/env python3
import math

def factorial(n):
  prod = 1
  for i in range(2, n+1):
    prod = prod * i
  
  return prod

# using Taylor series of arctan(1)
def pi_summation_slow(N):
  sum = 0.0
  sign = 1

  for i in range(0,N):
    sum = sum + sign/(2*i + 1.0)
    sign = -sign

  return 4*sum


# Using Euler's formula
# $$
# \arctan(x) = \sum_{n=0}^\infty \frac{2^{2n} (n!)^2}{(2n + 1)!} \frac{x^{2n + 1}}{(1 + x^2)^{n + 1}}
# $$
# and Machin's formula
# $$
# \frac{\pi}{4} = 4 \arctan\frac{1}{5} - \arctan\frac{1}{239}
# $$
def pi_summation_fast(N):
  sum = 0.0

  for n in range(0,N):
    f = factorial(n)
    common = math.pow(2.0, 2*n)*f*f/factorial(2*n + 1)
    A = math.pow(25/26, n+1)/math.pow(5, 2*n+1)
    B = math.pow(239*239 / (239*239 + 1), n+1)/math.pow(239, 2*n+1)
    sum = sum + common*( 4*A - B )

  return 4*sum;

# Taylor series approximation of sin(x)
# Each term is calculated explicitly
def taylor_sin(x, N):
  sum = 0.0
  sign = 1

  for n in range(0,N):
    sum = sum + sign*math.pow(x, 2*n + 1)/factorial(2*n + 1)
    sign = -sign

  return sum

# Optimised Taylor series approximation of sin(x)
# Each term is calculated from the previous one
def taylor_sin_opt(x, N):
  sum = x
  an = x

  for i in range(1,N):
    an = -x*x*an/(2*n*(2*n+1))
    sum = sum + an

  return sum


for n in range(1,100):
  print(n, taylor_sin(math.pi*(10+ 1/6.0), n))