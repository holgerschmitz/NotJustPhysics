#!/usr/bin/env python3
import math
import numpy as np

def isEven(n: int):
  return n % 2 == 0

def interpol_piecewise(data: np.ndarray, xmin: float, xmax: float):
  h = (xmax - xmin) / (data.shape[0] - 1)
  def interp_func(x):
    nn = round((x-xmin)/h)
    return data[nn]
  
  return interp_func

def interpol_linear(data: np.ndarray, xmin: float, xmax: float):
  L = data.shape[0] - 1
  h = (xmax - xmin) / L
  def interp_func(x):
    xp = x-xmin
    nn = math.floor(xp/h)
    if (nn==L):
      return data[nn]
    

    alpha = xp/h - nn
    return data[nn]*(1-alpha) + data[nn+1]*alpha
  
  return interp_func

def lagrange_polynomial(data: np.ndarray, xlo: float, h: float):
  N = data.shape[0]

  def interp_polynomial(x: float):
    sum = 0.0
    for i in range(N):
      prod = 1.0
      xi = xlo + i*h
      for j in range(N):
        if i==j:
          continue
        xj = xlo + j*h
        prod = prod * (x - xj)/((i-j)*h)
      sum = sum + prod
    return sum

def interpol_lagrange(data: np.ndarray, order: int, xmin: float, xmax: float):
  L = data.shape[0] - 1
  h = (xmax - xmin) / L
  oh = math.floor(order / 2)
  def interp_func(x):
    xp = x-xmin
    if (isEven(order)):
      nn = round(xp/h)
    else:
      nn = math.floor(xp/h)
    
    nstart = nn - oh
    if (nn==L):
      return data[nn]
    

    alpha = xp/h - nn
    return data[nn]*(1-alpha) + data[nn+1]*alpha
  
  return interp_func

def testFunc(x: np.ndarray):
  return np.sin(x + 0.1) + 2*np.cos(x/2) + 2*np.sin(1.5*x)

Ns = 20
Nt = 1000
xmin = 0.0
xmax = 2.2*math.pi
epst = 0.5*(xmax - xmin)/Nt

Xs = np.linspace(xmin, xmax, num=Ns+1)
Ds = testFunc(Xs)

Xtest = np.linspace(xmin + epst, xmax - epst, num=Nt)
F = testFunc(Xtest)

intFuncPW = interpol_piecewise(Ds, xmin, xmax)
intFuncLin = interpol_linear(Ds, xmin, xmax)

# for i in range(0, Ns):
#   print("{} {}".format(Xs[i], Ds[i]))

for i in range(0, Nt):
  print("{} {} {} {}".format(Xtest[i], F[i], intFuncPW(Xtest[i]), intFuncLin(Xtest[i])))