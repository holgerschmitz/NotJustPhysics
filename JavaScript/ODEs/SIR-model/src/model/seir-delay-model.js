var rk4 = require('ode-rk4');
 
export class SEIRDelayModel {
  constructor() {
    this.definition = {
      parameters: [
        { 
          name: 'beta',
          description: 'Infection Rate',
          min: 0.0,
          max: 10.0,
          step: 0.01
        },
        { 
          name: 'gamma',
          description: 'Recovery Rate',
          min: 0.0,
          max: 2.0,
          step: 0.01
        },
        { 
          name: 'inc',
          description: 'Incubation Period',
          min: 0.01,
          max: 10.0,
          step: 0.1
        }
      ],
      output: [
        {
          name: 'S'
        },
        {
          name: 'E'
        },
        {
          name: 'I'
        },
        {
          name: 'R'
        }
      ]
    };

    this.beta = 1;
    this.gamma = 0.5;
    this.inc = 1;
    this.tmax = 30;
    this.dt = 0.1;

    const Istart = 0.01;
    this.initialConditions = {
      S: 1-Istart,
      E: 0,
      I: Istart,
      R: 0
    };

    this.solution = [];
  }

  getRHS() {
    const beta = this.beta;
    const gamma = this.gamma;
    return (dydt, y, t) => {
      const d = this.index - this.delay;
      const yd = d<0 ? [1, 0, 0, 0] : this.solution[d];
      dydt[0] = -beta*y[0]*y[2];
      dydt[1] =  beta*y[0]*y[2]   - beta*yd[0]*yd[2];
      dydt[2] =  beta*yd[0]*yd[2] - gamma*y[2];
      dydt[3] =                     gamma*y[2];         
    }
  }

  integrate() {
    this.index = 0;
    this.delay = Math.floor(this.inc/this.dt);
    const y = [this.initialConditions.S, this.initialConditions.E, this.initialConditions.I, this.initialConditions.R];
    this.solution = [];
    this.solution.push([...y]);

    const integrator = rk4(y, this.getRHS(), 0, this.dt);
    while(integrator.t < this.tmax) {
      this.index++;
      integrator.step();
      this.solution.push([...integrator.y]);
    }
  }
}