var webpack = require('webpack');
var path = require('path');

var BUILD_DIR = path.resolve(__dirname, 'js');
var APP_DIR = path.resolve(__dirname, 'src');

var config = {
  entry: APP_DIR + '/index.ts',
  mode: 'production',
  output: {
    path: BUILD_DIR,
    filename: 'flockingjs.js',
    libraryTarget: 'var',
    library: 'FlockingJS'
  },
  resolve: {
    extensions: [ '.ts', '.js' ]
  },
  // devtool: 'inline-source-map',
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/
      }
    ]
  },
  optimization: {
    minimize: false
  }
};

module.exports = config;
