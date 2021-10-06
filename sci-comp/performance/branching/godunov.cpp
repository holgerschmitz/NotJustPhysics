#include <cmath>
#include <algorithm>
#include <iostream>

static const int N = 10000;
static const int nSteps = 10000;

// parameters
double g = 9.81;
double dx = 0.05;
double dt = 0.0001;

double h[N], q[N];
double dhdt[N], dqdt[N];
double zbf[N];
double hS[N], qS[N];
double dhdtS[N], dqdtS[N];

void init() {
  for(int i=0; i<N; i++) {
    h[i] = 1.0; 
    q[i] = 0.0;
    dhdt[i] = dqdt[i] = 0.0;
    zbf[i] = 0.0;
    hS[i] = qS[i] = 0.0;
    dhdtS[i] = dqdtS[i] = 0.0;
  }
  for(int i=N/2; i<N; i++) {
    h[i] = 0.1; 
  }
}

struct RiemannState {
  double L, R;
};

struct Flux {
  double h, q;
};

Flux hllc(RiemannState hFace, RiemannState qFace) {
  RiemannState u = {qFace.L/hFace.L, qFace.R/hFace.R};
  double h_star_p = (sqrt(g*hFace.L) + sqrt(g*hFace.R)) / 2.0 + (u.L - u.R) / 4.0;
  double u_star = (u.L + u.R) / 2.0 + sqrt(g*hFace.L) - sqrt(g*hFace.R);
  double h_star = (h_star_p*h_star_p) / g;

  RiemannState s = {
    std::min(u.L - sqrt(g*hFace.L), u_star - sqrt(g*h_star)),
    std::max(u.R + sqrt(g*hFace.R), u_star + sqrt(g*h_star))
  };

  RiemannState q_flux = {
    qFace.L*u.L + g*hFace.L*hFace.L / 2.0,
    qFace.R*u.R + g*hFace.R*hFace.R / 2.0
  };
  double h_flux_star = (s.R*qFace.L - s.L*qFace.R + s.L*s.R*(hFace.R - hFace.L)) / (s.R - s.L);
  double q_flux_star = (s.R*q_flux.L - s.L*q_flux.R + s.L*s.R*(qFace.R - qFace.L)) / (s.R - s.L);

  Flux f;

  if (s.L >= 0.0) {
    f.h = qFace.L;
    f.q = q_flux.L;
  } else if (s.R >= 0) {
    f.h = h_flux_star;
    f.q = q_flux_star;
  } else {
    f.h = qFace.R;
    f.q = q_flux.R;
  }

  return f;
}

Flux hllc_opt(RiemannState hFace, RiemannState qFace) {
  RiemannState u = {qFace.L/hFace.L, qFace.R/hFace.R};
  double h_star_p = (sqrt(g*hFace.L) + sqrt(g*hFace.R)) / 2.0 + (u.L - u.R) / 4.0;
  double u_star = (u.L + u.R) / 2.0 + sqrt(g*hFace.L) - sqrt(g*hFace.R);
  double h_star = (h_star_p*h_star_p) / g;

  RiemannState s = {
    std::min(u.L - sqrt(g*hFace.L), u_star - sqrt(g*h_star)),
    std::max(u.R + sqrt(g*hFace.R), u_star + sqrt(g*h_star))
  };

  RiemannState q_flux = {
    qFace.L*u.L + g*hFace.L*hFace.L / 2.0,
    qFace.R*u.R + g*hFace.R*hFace.R / 2.0
  };

  Flux f;

  if (s.L >= 0.0) {
    f.h = qFace.L;
    f.q = q_flux.L;
  } else if (s.R >= 0) {
    double h_flux_star = (s.R*qFace.L - s.L*qFace.R + s.L*s.R*(hFace.R - hFace.L)) / (s.R - s.L);
    double q_flux_star = (s.R*q_flux.L - s.L*q_flux.R + s.L*s.R*(qFace.R - qFace.L)) / (s.R - s.L);
    f.h = h_flux_star;
    f.q = q_flux_star;
  } else {
    f.h = qFace.R;
    f.q = q_flux.R;
  }

  return f;
}

Flux hllc_no_if(RiemannState hFace, RiemannState qFace) {
  RiemannState u = {qFace.L/hFace.L, qFace.R/hFace.R};
  double h_star_p = (sqrt(g*hFace.L) + sqrt(g*hFace.R)) / 2.0 + (u.L - u.R) / 4.0;
  double u_star = (u.L + u.R) / 2.0 + sqrt(g*hFace.L) - sqrt(g*hFace.R);
  double h_star = (h_star_p*h_star_p) / g;

  RiemannState s = {
    std::min(u.L - sqrt(g*hFace.L), u_star - sqrt(g*h_star)),
    std::max(u.R + sqrt(g*hFace.R), u_star + sqrt(g*h_star))
  };

  RiemannState q_flux = {
    qFace.L*u.L + g*hFace.L*hFace.L / 2.0,
    qFace.R*u.R + g*hFace.R*hFace.R / 2.0
  };
  double h_flux_star = (s.R*qFace.L - s.L*qFace.R + s.L*s.R*(hFace.R - hFace.L)) / (s.R - s.L);
  double q_flux_star = (s.R*q_flux.L - s.L*q_flux.R + s.L*s.R*(qFace.R - qFace.L)) / (s.R - s.L);

  Flux f[3];

  f[2].h = qFace.L;
  f[2].q = q_flux.L;
  f[1].h = h_flux_star;
  f[1].q = q_flux_star;
  f[0].h = qFace.R;
  f[0].q = q_flux.R;

  int case1 = s.L >= 0.0;
  int case2 = s.R >= 0;
  int index = 2*case1 + (1-case1)*case2;

  return f[index];
}

#define HLLC hllc

// function to calculate dhdt and dqdt
void calcDDt(double *h, double *q, double *dhdt, double *dqdt, double dx) {
  RiemannState h_west, h_right;
  RiemannState q_west, q_right;
  Flux flux_west, flux_east;

  // left boundary
  flux_west = HLLC(RiemannState{h[0], h[0]}, RiemannState{q[0], q[0]});
  flux_east = HLLC(RiemannState{h[0], h[1]}, RiemannState{q[0], q[1]});

  dhdt[0] = - (flux_east.h - flux_west.h)/dx;
  dqdt[0] = - (flux_east.q - flux_west.q)/dx;

  // right boundary
  flux_west = HLLC(RiemannState{h[N-2], h[N-1]}, RiemannState{q[N-2], q[N-1]});
  flux_east = HLLC(RiemannState{h[N-1], h[N-1]}, RiemannState{q[N-1], q[N-1]});

  dhdt[N-1] = - (flux_east.h - flux_west.h)/dx;
  dqdt[N-1] = - (flux_east.q - flux_west.q)/dx;

  // volume
  for (int i=1; i<N-1; ++i) {
    flux_west = hllc(RiemannState{h[i-1], h[i]}, RiemannState{q[i-1], q[i]});
    flux_east = hllc(RiemannState{h[i], h[i+1]}, RiemannState{q[i], q[i+1]});

    dhdt[i] = - (flux_east.h - flux_west.h)/dx;
    dqdt[i] = - (flux_east.q - flux_west.q)/dx;
  }
}

int main() {
  init();
  for (int t=0; t<nSteps; ++t) {
    calcDDt(h, q, dhdt, dqdt, dx);
    for (int i=0; i<N; ++i) {
      hS[i] = h[i] + dhdt[i]*dt;
      qS[i] = q[i] + dqdt[i]*dt;
    }

    calcDDt(hS, qS, dhdtS, dqdtS, dx);
    for (int i=0; i<N; ++i) {
      h[i] = hS[i] + (dhdt[i] + dhdtS[i])*dt/2.0;
      q[i] = qS[i] + (dqdt[i] + dqdtS[i])*dt/2.0;
    }
  }
  // for (int i=0; i<N; ++i) {
  //   std::cout << i << " " << h[i] << " " << q[i] << std::endl;
  // }
}


     



