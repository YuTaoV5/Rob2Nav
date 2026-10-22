import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
const source=await readFile(new URL('../src/composerPresentation.ts',import.meta.url),'utf8');
const js=ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.ESNext}}).outputText;
const {planTiming,timingDisplay}=await import('data:text/javascript;base64,'+Buffer.from(js).toString('base64'));
const plan=[{seconds:2,trigger:false},{seconds:3,trigger:false},{seconds:2,trigger:false}];
test('planned duration follows the backend shared-stand cycle expansion',()=>{
 assert.deepEqual(planTiming(plan,1),{count:3,dwell:7,atomic:0});
 assert.deepEqual(planTiming(plan,2),{count:5,dwell:12,atomic:0});
 assert.deepEqual(planTiming(plan,4),{count:9,dwell:22,atomic:0});
});
test('atomic actions are counted separately from dwell',()=>{
 assert.deepEqual(planTiming([plan[0],{seconds:99,trigger:true},plan[2]],1),{count:3,dwell:4,atomic:1});
});
test('rounded displayed parts always add to the displayed total',()=>{
 assert.deepEqual(timingDisplay({total_elapsed_s:3.08,dwell_elapsed_s:2.04}),{total:'3.1',dwell:'2.0',extra:'1.1'});
});
