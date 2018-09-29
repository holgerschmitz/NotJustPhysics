// import * as THREE from 'three';
// declare global {
//   const THREE:any;
// }

function makeTriangle(height:number) {
  var shape = new THREE.Shape();
  shape.moveTo( -1, 0 );
  shape.lineTo(  1, 0 );
  shape.lineTo(  0, height );
  shape.lineTo( -1, 0 );

  var extrudeSettings = {
    steps: 1,
    amount: 0.01,
    bevelEnabled: false
  };

  var geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings);
  return geometry;
};

export class Bird {
  public bird: THREE.Object3D = new THREE.Object3D();
  private leftWing: THREE.Mesh;
  private rightWing: THREE.Mesh;
  private body: THREE.Mesh;
  private wrapper: THREE.Object3D = new THREE.Object3D();

  private flapAmplitude: number;
  private up: THREE.Vector3;
  private zero: THREE.Vector3;

  public flapAngle: number;
  public velocity: THREE.Vector3;
  public posref: {x:number, y:number, z:number, radius:number, id:number};
  constructor(material:THREE.MeshBasicMaterial, flap:number, id:number) {
    this.leftWing = new THREE.Mesh(makeTriangle(3), material);
    this.leftWing.rotation.order = 'YXZ';
    this.leftWing.rotation.x = 0.5*Math.PI;
    this.leftWing.rotation.y = 0.5*Math.PI;
    this.leftWing.rotation.z = 0.0;

    this.rightWing = new THREE.Mesh(makeTriangle(3), material);
    this.rightWing.rotation.order = 'YXZ';
    this.rightWing.rotation.x = -0.5*Math.PI;
    this.rightWing.rotation.y = 0.5*Math.PI;
    this.rightWing.rotation.z = 0.0;

    this.body = new THREE.Mesh(makeTriangle(10), material);
    this.body.rotateY(0.5*Math.PI);
    this.body.rotateZ(0.5*Math.PI);
    this.body.translateY(-2);
    this.body.scale.setScalar(0.7);

    this.bird.add(this.leftWing);
    this.bird.add(this.rightWing);
    this.bird.add(this.body);
    this.bird.scale.setScalar(0.3);

    this.wrapper.add(this.bird);

    this.flapAngle = 0.0;
    this.flapAmplitude = flap;

    this.velocity = new THREE.Vector3(0,0,0);
    this.zero = new THREE.Vector3(0,0,0);
    this.up = new THREE.Vector3(0,1,0);
    this.posref = {x: this.wrapper.position.x, y: this.wrapper.position.y, z: this.wrapper.position.z, radius: 0.01, id: id};
  }

  stepTime(dt:number):void {
    this.flapAngle += dt;
    if (this.flapAngle > 2*Math.PI) this.flapAngle -= 2*Math.PI;
    var rot = this.flapAmplitude * Math.cos(this.flapAngle);
    this.rightWing.rotation.x = -0.5*Math.PI + rot;
    this.leftWing.rotation.x = 0.5*Math.PI - rot;
    this.bird.position.y = -0.3*rot;

    this.velocity.clampLength(0.8,1.2);

    this.wrapper.position.x += dt*this.velocity.x;
    this.wrapper.position.y += dt*this.velocity.y;
    this.wrapper.position.z += dt*this.velocity.z;
    this.pointForward();
    this.updateRef();
  }

  addToScene(scene:THREE.Scene):void {
    scene.add(this.wrapper);
  }

  object():THREE.Object3D {
    return this.wrapper;
  }

  pointForward():void {
    var look = new THREE.Vector3(0,0,0);
    look.add(this.velocity).normalize().add(this.wrapper.position);
    this.wrapper.lookAt(look);
  }

  applyForce(dt:number, force:THREE.Vector3):void {
    this.velocity.x += dt*force.x;
    this.velocity.y += dt*force.y;
    this.velocity.z += dt*force.z;
  }

  updateRef():void {
    this.posref.x = this.wrapper.position.x;
    this.posref.y = this.wrapper.position.y;
    this.posref.z = this.wrapper.position.z;
  }
}
