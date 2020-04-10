import React, { Component } from "react";

import { SIRModel } from '../model/sir-model';
import { SEIRModel } from '../model/seir-model';
import { Model } from './Model';

import InputLabel from '@material-ui/core/InputLabel';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import Select from '@material-ui/core/Select';

export class App extends Component {
  constructor() {
    super();
    this.state = {
      model: 'sir',
    }
  }

  handleModelChange(event) {
    this.setState({
      model: event.target.value,
      modelFactory: this.getModelFactory(event.target.value)
    });
  }

  getModelFactory(model) {
    switch (model) {
      case 'seir': 
        return () => new SEIRModel();
      case 'sir':
      default:
        return () => new SIRModel();
    }
  }

  getModelComponent(model) {
    if (this.state.model === model) {
      return (<Model model={this.getModelFactory(model)} />)
    }
    return (<span />);
  }

  render() {
    return (
      <div>
        <h3>Modeling Epidemics</h3>
        <FormControl>
          <InputLabel id="model-select-label">Model</InputLabel>
          <Select
            labelId="model-select-label"
            id="model-select"
            value={this.state.model}
            onChange={this.handleModelChange.bind(this)}
          >
            <MenuItem value='sir'>SIR Model</MenuItem>
            <MenuItem value='seir'>SEIR Model</MenuItem>
          </Select>
        </FormControl>
        {this.getModelComponent('sir')}
        {this.getModelComponent('seir')}
      </div>
    );
  }
}
