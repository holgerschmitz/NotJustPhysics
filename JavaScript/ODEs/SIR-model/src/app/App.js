import React, { Component } from "react";
import Slider from '@material-ui/core/Slider';
import { Chart } from 'react-charts'

import { SIRModel } from '../model/sir-model';

export class App extends Component {
  constructor() {
    super();

    this.model = new SIRModel();
    this.model.tmax = 30;
    this.model.dt = 0.05;

    this.model.integrate();

    this.state = {
      data: this.makeChartData(this.model.solution),
    };

    for (const parameter of this.model.definition.parameters) {
      this.state[parameter.name] = this.model[parameter.name];
    }
  }

  integrateModel() {
    this.model.beta = this.state.beta;
    this.model.gamma = this.state.gamma;
    this.model.integrate(this.updateData.bind(this));
  }

  handleParameterChange(name, event, newValue) {
    const newState = {};
    newState[name] = newValue;
    this.setState(newState);
    
    this.model[name] = newValue;
    this.model.integrate();
    this.updateData();
  }

  makeChartData(solution) {
    const data = this.model.definition.output.map(spec => (
      {
        label: spec.name,
        data: []
      }
    ));
    let it = 0;
    const n = this.model.definition.output.length;
    for (const y of solution) {
      if (it %10 === 0) {
        const t = it*this.model.dt;
        for (let i=0; i<n; i++) {
          data[i].data.push([t, y[i]]);
        }
      }
      it++;
    }
    return data;
  }

  updateData() {
    this.setState({
      data: this.makeChartData(this.model.solution)
    });
  }

  makeSliders() {
    return this.model.definition.parameters.map(parameter => (
      <label>{parameter.description}: {this.state[parameter.name]}
        <Slider 
          value={this.state[parameter.name]} 
          onChange={this.handleParameterChange.bind(this, parameter.name)} 
          min={parameter.min}
          step={parameter.step}
          max={parameter.max}
          aria-labelledby="continuous-slider" />
      </label>
    ));
  }

  render() {
    const axes = [
      { primary: true, type: 'linear', position: 'bottom' },
      { type: 'linear', position: 'left' }
    ];
    const sliders = this.makeSliders();
    return (
      <div>
        <h3>The SIR Model for the spread of infectious diseases</h3>
        {sliders}
        <div  style={{
          width: '100%',
          height: '500px'
        }}>
          <Chart data={this.state.data} axes={axes} />
        </div>
      </div>
    );
  }
}
