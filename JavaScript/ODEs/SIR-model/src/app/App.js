import React, { Component } from "react";

import { SIRModel } from '../model/sir-model';
import { SEIRModel } from '../model/seir-model';
import { SEIRDelayModel } from '../model/seir-delay-model';
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
      case 'seir-delay': 
        return () => new SEIRDelayModel();
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
            <MenuItem value='seir-delay'>SEIR Model with Delay</MenuItem>
          </Select>
        </FormControl>
        <h4>Model Parameters</h4>
        {this.getModelComponent('sir')}
        {this.getModelComponent('seir')}
        {this.getModelComponent('seir-delay')}
      </div>
    );
  }
}
