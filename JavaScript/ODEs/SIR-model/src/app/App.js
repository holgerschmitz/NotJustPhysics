import React, { Component } from "react";
import Slider from '@material-ui/core/Slider';
import { Chart } from 'react-charts'

import { SIRModel } from '../model/sir-model';

export class App extends Component {
  constructor() {
    super();

    this.handleChangeBeta = this.handleChangeBeta.bind(this);
    this.handleChangeGamma = this.handleChangeGamma.bind(this);
    this.model = new SIRModel();
    this.model.tmax = 30;
    this.model.dt = 0.05;
    this.model.beta = 1;
    this.model.gamma = 0.5;

    this.model.integrate();

    this.state = {
      beta: this.model.beta,
      gamma: this.model.gamma,
      data: this.makeChartData(this.model.solution),
    };
  }

  integrateModel() {
    this.model.beta = this.state.beta;
    this.model.gamma = this.state.gamma;
    this.model.integrate(this.updateData.bind(this));
  }

  handleChangeBeta(event, newValue) {
    this.setState(() => ({
        beta: newValue
    }));
    this.model.beta = newValue;
    this.model.integrate();
    this.updateData();
  }

  handleChangeGamma(event, newValue) {
    this.setState(() => ({
      gamma: newValue
    }));
    this.model.gamma = newValue;
    this.model.integrate();
    this.updateData();
  }

  makeChartData(solution) {
    const data = [
      {
        label: 'S',
        data: []
      },
      {
        label: 'I',
        data: []
      },
      {
        label: 'R',
        data: []
      },
    ];
    let t=0;
    let it = 0;
    for (const y of solution) {
      if (it %10 === 0) {
        data[0].data.push([t, y[0]]);
        data[1].data.push([t, y[1]]);
        data[2].data.push([t, y[2]]);
      }
      t += this.model.dt - 1e-9;
      it++;
    }
    return data;
  }

  updateData() {
    this.setState({
      data: this.makeChartData(this.model.solution)
    });
  }

  render() {
    const axes = [
      { primary: true, type: 'linear', position: 'bottom' },
      { type: 'linear', position: 'left' }
    ];
    return (
      <div>
        <h3>The SIR Model for the spread of infectious diseases</h3>
        <label>Infection Rate: {this.state.beta}
          <Slider 
            value={this.state.beta} 
            onChange={this.handleChangeBeta} 
            min={0.0}
            step={0.01}
            max={10}
            aria-labelledby="continuous-slider" />
        </label>
        <label>Recovery Rate: {this.state.gamma}
          <Slider 
            value={this.state.gamma} 
            onChange={this.handleChangeGamma} 
            min={0.0}
            step={0.01}
            max={10}
            aria-labelledby="continuous-slider" /> 
        </label>
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
