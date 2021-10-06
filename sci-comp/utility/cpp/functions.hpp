inline double factorial(int n)
{
  double result = 1.0;
  for (int i=2; i<=n; ++i) 
  {
    result *= i;
  }
  return result;
}

inline double ipow(int n)
{
  double result = 1.0;
  for (int i=2; i<=n; ++i) 
  {
    result *= i;
  }
  return result;
}