// import * as THREE from 'three';
// declare global {
//   const THREE:any;
// }

import {Bird} from './bird';

const LENGTH = 20;

export class Flock {
  private flock: Array<Bird> = [];
  private n: number;
  private time: number = 0.0;
  private octree: THREE.Octree;

  constructor(n:number, material:THREE.MeshBasicMaterial, scene:THREE.Scene) {
    this.n = n;

    this.octree = new THREE.Octree({
        radius: 1,
        undeferred: false,
        // set the max depth of tree
        depthMax: Infinity,
        // max number of objects before nodes split or merge
        objectsThreshold: 8,
        // percent between 0 and 1 that nodes will overlap each other
        // helps insert objects that lie over more than one node
        overlapPct: 0.15,
        // pass the scene to visualize the octree
        // scene: scene
      } );

    for (var i=0; i<n; i++) {
      var bird = new Bird(material, 0.6, i);
      var bo = bird.object();
      bo.position.x = LENGTH*(Math.random() * 2 - 1);
      bo.position.y = LENGTH*(Math.random() * 2 - 1);
      bo.position.z = LENGTH*(Math.random() * 2 - 1);
      bo.scale.setScalar(0.5);

      bird.velocity.x = Math.random() * 2 - 1;
      bird.velocity.y = Math.random() * 2 - 1;
      bird.velocity.z = Math.random() * 2 - 1;
      bird.velocity.normalize();

      bird.flapAngle = 2*Math.PI*Math.random();
      this.flock[i] = bird;

      bird.pointForward();
      bird.updateRef();
    }
  }

  addToScene(scene:THREE.Scene):void {
    for (var i=0; i<this.n; i++) {
      this.flock[i].addToScene(scene);
      this.octree.add(this.flock[i].posref);
    }
    this.octree.update();
  }

  stepTime(dt:number):void {
    for (var i=0; i<this.n; i++) this.octree.remove(this.flock[i].posref);
    for (var i=0; i<this.n; i++) this.flock[i].stepTime(dt);
  //  for (var i=0; i<this.n; i++) this.flock[i].updateRef();
    for (var i=0; i<this.n; i++) this.octree.add(this.flock[i].posref);
    this.octree.update();
    this.applyForces(dt);
  }

  applyForces(dt:number):void {
    this.time += dt;
    for (var i=0; i<this.n; i++) {
      var up = new THREE.Vector3(0,1,0);
      var bird = this.flock[i];
      var pos = bird.object().position;
      if (pos.x<-LENGTH) bird.applyForce(0.02*dt, new THREE.Vector3(-LENGTH-pos.x,0,0));
      if (pos.x>LENGTH) bird.applyForce(0.02*dt, new THREE.Vector3(LENGTH-pos.x,0,0));
      if (pos.y<-LENGTH) bird.applyForce(0.02*dt, new THREE.Vector3(0,-LENGTH-pos.y,0));
      if (pos.y>LENGTH) bird.applyForce(0.02*dt, new THREE.Vector3(0,LENGTH-pos.y,0));
      if (pos.z<-LENGTH) bird.applyForce(0.02*dt, new THREE.Vector3(0,0,-LENGTH-pos.z));
      if (pos.z>LENGTH) bird.applyForce(0.02*dt, new THREE.Vector3(0,0,LENGTH-pos.z));
      bird.applyForce(0.2*dt, up.clone().multiplyScalar(Math.sin(0.1*pos.x)*Math.sin(0.1*pos.y)*Math.sin(0.1*this.time)));
      var neighbours = this.octree.search(pos, 10.0, true, new THREE.Vector3(1,0,0));
      var avgV = bird.velocity.clone();
      for (var j=0; j<neighbours.length; j++) {
        var n = neighbours[j].object;
        var b = this.flock[n.id];
        var r = new THREE.Vector3(n.x, n.y, n.z);
        r.sub(pos);

        var p = b.velocity.clone().cross(up).cross(b.velocity).normalize();
        var pdist = p.clone().dot(r);
        var fw = b.velocity.clone().dot(r);
        var l = r.length()/5;
        if ((l>2.0) || (n.id==i) || (fw<0)) continue;
        if (l<1.0) l=2*l-1;
        else l = 1/(l*l) - 1;
        r.normalize().multiplyScalar(l);
        bird.applyForce(0.1*dt,r);

        bird.applyForce(0.01*dt, p.multiplyScalar(pdist));

        avgV.add(b.velocity);

      }
      avgV.normalize().sub(bird.velocity);
      bird.applyForce(0.1*dt,avgV);
      bird.applyForce(0.01*dt,new THREE.Vector3(Math.random()-0.5,Math.random()-0.5,Math.random()-0.5));
    }
  }

  applyRay(rayCaster) {
    const o = rayCaster.ray.origin.clone();
    const d = rayCaster.ray.direction.clone().normalize();
    var affected = this.octree.search(o, 30, true, d);

    for (var i=0; i<affected.length; i++) {
      const obj = affected[i].object;
      const r = new THREE.Vector3(obj.x, obj.y, obj.z);
      r.sub(o);
      const rho = r.clone().sub(r.clone().projectOnVector(d));
      const l = rho.length();
      this.flock[obj.id].applyForce(0.1, rho.setLength(10));
    }
  }
}
