
import {Flock} from './flock';

export class MouseForce {
  private flock:Flock;
  private camera;
  constructor(flock:Flock, camera) {
    this.flock = flock;
    this.camera = camera;

    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);

    document.addEventListener('mousedown', this.onMouseDown, false);
  }

  onMouseDown(event:MouseEvent):void {
    if (event.button === THREE.MOUSE.RIGHT) {
      document.addEventListener('mousemove', this.onMouseMove, false);
      document.addEventListener('mouseup', this.onMouseUp, false);
    }
  }

  onMouseUp(event:MouseEvent):void {
    document.removeEventListener('mousemove', this.onMouseMove, false );
    document.removeEventListener('mouseup', this.onMouseUp, false);
  }

  onMouseMove(event) {

  	var mousePosition = new THREE.Vector3();
  	mousePosition.x = (event.pageX / window.innerWidth) * 2 - 1;
  	mousePosition.y = -(event.pageY / window.innerHeight) * 2 + 1;
  	mousePosition.z = 0.5;

  	mousePosition.unproject(this.camera);

  	var origin = new THREE.Vector3().copy(this.camera.position);
  	var direction = new THREE.Vector3().copy(mousePosition.sub(this.camera.position)).normalize();
  	var rayCaster = new THREE.Raycaster(origin, direction);

    this.flock.applyRay(rayCaster);
  }

}
