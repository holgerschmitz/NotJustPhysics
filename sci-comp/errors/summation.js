#!/usr/bin/env node

function factorial(n) {
  let prod = 1;
  for (i=2; i<=n; i++) {
    prod *= i;
  }
  return prod;
}

// using Taylor series of arctan(1)
function pi_summation_slow(N) {
  let sum = 0.0;
  let sign = 1;

  for (let i=0; i<N; ++i) {
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
function pi_summation_fast_overflow(N) {
  let sum = 0.0;

  for (let n=0; n<N; ++n) {
    const f = factorial(n);
    const common = Math.pow(2.0, 2*n)*f*f/factorial(2*n + 1);
    const A = pow(25/26, n+1)/pow(5, 2*n+1);
    const B = pow(239*239 / (239*239 + 1), n+1)/pow(239, 2*n+1);
    sum += common*( 4*A - B );
  }

  return 4*sum;
}

// Taylor series approximation of sin(x)
// Each term is calculated explicitly
function taylor_sin(x, N) {
  let sum = 0.0;
  let sign = 1;

  for (let n=0; n<N; n++) {
    sum += sign*pow(x, 2*n + 1)/factorial(2*n +1);
    sign = -sign;
  }

  return sum;
}

// Optimised Taylor series approximation of sin(x)
// Each term is calculated from the previous one
function taylor_sin_opt(x, N) {
  let sum = x;
  let an = x;

  for (let n=1; n<N; n++) {
    an = -x*x*an/(2*n*(2*n+1));
    sum += an;
  }

  return sum;
}


const N = 10000;
console.log(N, math.fabs(pi_summation_slow(N)) - Math.PI);
