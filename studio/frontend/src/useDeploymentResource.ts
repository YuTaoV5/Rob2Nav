import {useEffect,useState} from 'react';

/** Retry read-only metadata after a tunnel/server reconnect. Never retry mutations. */
export default function useDeploymentResource<T>(path:string){
 const [data,setData]=useState<T>(),[error,setError]=useState('');
 useEffect(()=>{
  let active=true;let timer:ReturnType<typeof setTimeout>|undefined;
  const controller=new AbortController();
  async function load(){
   try{
    const response=await fetch(path,{cache:'no-store',signal:controller.signal});
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const result=await response.json();
    if(active){setData(result);setError('')}
   }catch(e){
    if(active){setError('部署服务连接暂时中断，正在自动重连…');timer=setTimeout(load,3000)}
   }
  }
  load();
  return()=>{active=false;controller.abort();if(timer)clearTimeout(timer)};
 },[path]);
 return {data,error};
}
