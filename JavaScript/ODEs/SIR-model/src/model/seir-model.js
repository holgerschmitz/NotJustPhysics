var rk4 = require('ode-rk4');
 
export class SEIRModel {
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
          name: 'a',
          description: 'Incubation Rate (inverse of the average incubation period)',
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
    this.a = 1;
    this.tmax = 30;
    this.dt = 0.1;

    const Estart = 0.01;
    this.initialConditions = {
      S: 1-Estart,
      E: Estart,
      I: 0,
      R: 0
    };

    this.solution = [];
  }

  getRHS() {
    const beta = this.beta;
    const gamma = this.gamma;
    const a = this.a;
    return (dydt, y, t) => {
      dydt[0] = -beta*y[0]*y[2];
      dydt[1] =  beta*y[0]*y[2] - a*y[1];
      dydt[2] =  a*y[1] - gamma*y[2];
      dydt[3] =           gamma*y[2];      
    }
  }

  integrate() {
    const y = [this.initialConditions.S, this.initialConditions.E, this.initialConditions.I, this.initialConditions.R];
    this.solution = [];
    this.solution.push([...y]);

    const integrator = rk4(y, this.getRHS(), 0, this.dt);
    while(integrator.t < this.tmax) {
      integrator.step();
      this.solution.push([...integrator.y]);
    }
  }
}