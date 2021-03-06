var rk4 = require('ode-rk4');
 
export class SIRModel {
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
        }
      ],
      output: [
        {
          name: 'Susceptible'
        },
        {
          name: 'Infected'
        },
        {
          name: 'Recovered'
        }
      ]
    };

    this.beta = 1;
    this.gamma = 0.5;
    this.tmax = 30;
    this.dt = 0.1;

    const Istart = 0.01;
    this.initialConditions = {
      S: 1-Istart,
      I: Istart,
      R: 0
    };

    this.solution = [];
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
    this.solution.push([...y]);

    const integrator = rk4(y, this.getRHS(), 0, this.dt);
    while(integrator.t < this.tmax) {
      integrator.step();
      this.solution.push([...integrator.y]);
    }
  }
}