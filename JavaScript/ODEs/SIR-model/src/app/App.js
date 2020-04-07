import React, { Component } from "react";
import Slider from '@material-ui/core/Slider';
import Plot from 'react-plotly.js';

import { SIRModel } from '../model/sir-model';

export class App extends Component {
  constructor() {
    super();

    this.state = {
      beta: 1,
      gamma: 0.5
    };

    this.handleChangeBeta = this.handleChangeBeta.bind(this);
    this.handleChangeGamma = this.handleChangeGamma.bind(this);
    this.model = new SIRModel();
  }

  integrateModel() {
    this.model.beta = this.state.beta;
    this.model.gamma = this.state.gamma;
    this.model.integrate();
  }

  handleChangeBeta(event, newValue) {
    this.setState(() => ({
        beta: newValue
    }));
  }

  handleChangeGamma(event, newValue) {
    this.setState(() => ({
      gamma: newValue
    }));
  }

  render() {
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
        <Plot
        data={[
          {
            x: [1, 2, 3],
            y: [2, 6, 3],
            type: 'scatter',
            mode: 'lines+markers',
            marker: {color: 'red'},
          },
          {type: 'bar', x: [1, 2, 3], y: [2, 5, 3]},
        ]}
        layout={{width: 320, height: 240, title: 'A Fancy Plot'}}
      />
      </div>
    );
  }
}
