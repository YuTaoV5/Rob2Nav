import {useEffect,useRef,useState} from 'react';
import * as THREE from 'three';
import {OBJLoader} from 'three/addons/loaders/OBJLoader.js';
import URDFLoader,{URDFRobot} from 'urdf-loader';
import {createViewport} from './RobotViewer';

type Obj=Record<string,any>;
export default function ReplayScene({scene,frame}:{scene:Obj;frame?:Obj}){
 const host=useRef<HTMLDivElement>(null),robot=useRef<URDFRobot|null>(null),view=useRef<ReturnType<typeof createViewport>|null>(null),latest=useRef(frame),[error,setError]=useState(''),[follow,setFollow]=useState(true);const followRef=useRef(follow);followRef.current=follow;latest.current=frame;
 useEffect(()=>{if(!host.current)return;let dead=false,v:ReturnType<typeof createViewport>;setError('');
 try{v=createViewport(host.current,[3.5,2.5,3.5],[0,.2,0]);view.current=v;v.grid.visible=false;const world=new THREE.Group();world.rotation.x=-Math.PI/2;v.scene.add(world);
 const material=new THREE.MeshStandardMaterial({color:0x8aade8,roughness:.9});
 (scene.boxes||[]).forEach((b:number[])=>{const [x0,x1,y0,y1,top,bottom]=b,o=scene.offset||[-4,-4,0];const m=new THREE.Mesh(new THREE.BoxGeometry(x1-x0,y1-y0,top-bottom),material);m.position.set((x0+x1)/2+o[0],(y0+y1)/2+o[1],(top+bottom)/2+o[2]);world.add(m)});
 (scene.geometries||[]).forEach((g:Obj)=>{const s=g.size;let geometry:THREE.BufferGeometry;if(g.type===6)geometry=new THREE.BoxGeometry(s[0]*2,s[1]*2,s[2]*2);else if(g.type===5){geometry=new THREE.CylinderGeometry(s[0],s[0],s[1]*2,32);geometry.rotateX(Math.PI/2)}else geometry=new THREE.PlaneGeometry(60,30);const mesh=new THREE.Mesh(geometry,material);mesh.position.fromArray(g.position);const q=g.quaternion;mesh.quaternion.set(q[1],q[2],q[3],q[0]);world.add(mesh)});
 const manager=new THREE.LoadingManager();manager.setURLModifier(url=>{if(!url.startsWith('/api/robot/assets/'))throw Error('不允许外部网格');return url});manager.onError=url=>!dead&&setError('网格加载失败：'+url);
 const loader=new URDFLoader(manager);loader.loadMeshCb=(url,m,material,done)=>new OBJLoader(m).load(url,o=>{o.traverse(n=>{if((n as THREE.Mesh).isMesh)(n as THREE.Mesh).material=material});done(o)},undefined,e=>done(new THREE.Group(),Error(String(e))));
 fetch('/api/robot/go2.urdf').then(r=>r.text()).then(xml=>{if(dead)return;const r=loader.parse(xml);Object.values(r.joints).forEach(j=>j.ignoreLimits=true);robot.current=r;world.add(r)}).catch(e=>!dead&&setError(String(e)));
 }catch(e){setError(String(e));return}
 let focused=false;const previous=new THREE.Vector3();const timer=setInterval(()=>{const f=latest.current,r=robot.current;if(!r||!f?.joints||!f?.quaternion)return;r.position.fromArray(f.position);const q=f.quaternion;r.quaternion.set(q[1],q[2],q[3],q[0]);if(!focused){const p=new THREE.Vector3(f.position[0],f.position[2],-f.position[1]);v.camera.position.copy(p).add(new THREE.Vector3(2.5,1.8,2.5));v.controls.target.copy(p);v.controls.update();focused=true;previous.copy(p)}else{const p=new THREE.Vector3(f.position[0],f.position[2],-f.position[1]);if(followRef.current){const delta=p.clone().sub(previous);v.camera.position.add(delta);v.controls.target.add(delta)}previous.copy(p)}(scene.joint_names||[]).forEach((name:string,i:number)=>r.setJointValue(name,f.joints[i]));},30);
 return()=>{dead=true;clearInterval(timer);robot.current=null;v.dispose()};
 },[scene]);
 return <div className="replay-3d"><div ref={host} className="viewport" aria-label="真实关节状态三维仿真视图"/>{error&&<div className="alert danger">{error}</div>}<div className="viewport-toolbar"><span>左键拖动旋转 · 右键拖动平移 · 滚轮缩放</span><label><input type="checkbox" checked={follow} onChange={e=>setFollow(e.target.checked)}/>跟随机器狗</label><button className="button quiet" onClick={()=>{const f=latest.current,v=view.current;if(!v||!f)return;const p=new THREE.Vector3(f.position[0],f.position[2],-f.position[1]);v.camera.position.copy(p).add(new THREE.Vector3(2.5,1.8,2.5));v.controls.target.copy(p);v.controls.update()}}>定位机器狗</button><button className="button quiet" onClick={()=>view.current?.reset()}>重置视角</button></div></div>
}
