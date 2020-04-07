import React from "react";
import ReactDOM from "react-dom";
import { App } from './app/App';


jQuery(document).ready(() => {
  const wrapper = document.getElementById("sir-model-container");
  wrapper ? ReactDOM.render(<App />, wrapper) : false;
});