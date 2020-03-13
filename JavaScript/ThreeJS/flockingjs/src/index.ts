// import * as THREE from 'three';
declare global {
  var THREE;
}

import {Flock} from './flock';
import {MouseForce} from './mouseforce';
import {OrbitControls} from './OrbitControls';

var scene = new THREE.Scene();
var camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);

var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0xffffff, 1);
document.body.appendChild(renderer.domElement);

var geometry = new THREE.BoxGeometry(1, 1, 1);
var material = new THREE.MeshPhongMaterial({color: 0x00ff00});

var flock = new Flock(200, material, scene);

flock.addToScene(scene);

var ambientLight = new THREE.AmbientLight(0x222222);
scene.add(ambientLight);

var lights = [];
lights[0] = new THREE.PointLight(0xffffff, 1, 0);
lights[1] = new THREE.PointLight(0xffffff, 1, 0);
lights[2] = new THREE.PointLight(0xffffff, 1, 0);

lights[0].position.set(0, 200, 0);
lights[1].position.set(100, 200, 100);
lights[2].position.set(- 100, - 200, - 100);

scene.add(lights[0]);
scene.add(lights[1]);
scene.add(lights[2]);

var controls = new OrbitControls(camera);
camera.position.z = 50;
controls.update();

new MouseForce(flock, camera);

function render() {
  flock.stepTime(0.1);
  requestAnimationFrame(render);
  controls.update();
  renderer.render(scene, camera);
  flock.stepTime(0.05);
}

render();
