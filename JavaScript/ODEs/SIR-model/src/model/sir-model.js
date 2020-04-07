var rk4 = require('ode-rk4')
 
export class SIRModel {
  constructor() {
    this.beta = 1;
    this.gamma = 1;
    this.tmax = 30;
    this.dt = 0.01;

    const Istart = 0.01;
    this.initialConditions = {
      S: 1-Istart,
      I: Istart,
      R: 0
    };

    this.solution = [];
    this.callbacks = [];
  }

  onDone(cb) {
    this.callbacks.push(cb);
  }

  getRHS() {
    const beta = this.beta;
    const gamma = this.gamma;
    return (dydt, y, t) => {
      dydt[0] = -beta*y[0]*y[1];
      dydt[1] =  beta*y[0]*y[1] - gamma*y[1];
      dydt[2] =                   gamma*y[1];      
    }
  }

  integrate() {
    const y = [this.initialConditions.S, this.initialConditions.I, this.initialConditions.R];
    this.solution = [];
    this.solution.push(y);

    const integrator = rk4(y, this.getRHS(), 0, this.dt);
    while(integrator.t < this.tmax) {
      integrator.step();
      this.solution.push(integrator.y);
    }

    for (cb of this.callbacks) {
      cb(this.solution);
    }
  }
}