/* Data transforms for the model-analysis workbench. No source record is modified. */
(function(root){
'use strict';
const node=typeof module!=='undefined' && module.exports;
const A=node?require('./guided-data.js'):root.NpuDemoData;
const C=node?require('./cost-model.js'):{estimate:root.estimate};
const EXPERIMENTS=[
 ['e1_baselines','E1','모델 기준선','모델','FP32 정확도, 모델 규모 및 비용 추정'],
 ['e2_ptq_grid','E2','양자화 방식 비교','양자화','7개 PTQ 방식의 정확도와 정수 출력 비교'],
 ['e3_lint_vs_drop','E3','정적 진단과 양자화 손실','양자화','진단 지표와 계층별 손실의 관계'],
 ['e4_surgery','E4','모델 변경·재학습','모델','활성함수 교체, CLE, 바이어스 보정 및 QAT'],
 ['e5_calibration','E5','캘리브레이션 표본','양자화','보정 이미지 수와 표집 조건별 정확도'],
 ['e6_pruning','E6','채널 프루닝','모델','MACs, 정확도 및 배열 정렬에 따른 추정 비용'],
 ['e7_requant_ablation','E7','정수 연산 구성','양자화','승수·누산기·바이어스 비트폭과 반올림 비교'],
 ['e8_scalesim','E8','SCALE-Sim 대조','비용·외부 검증','동일 이상화 조건의 연산 사이클 대조'],
 ['e9_customer_intake','E9','모델 분석·변경 비교','모델','지원 연산, 비용 추정 및 모델 변경 기록'],
 ['e10_engine_timing','E10','호스트 실행 시간','비용·외부 검증','NumPy·C++ 정수 검증 엔진의 CPU 실행 시간'],
 ['e11_tflite_crosscheck','E11','TFLite 참조 커널 대조','비용·외부 검증','conv·MEAN·fully-connected의 출력 일치 검사'],
 ['e12_imagenette','E12','Imagenette 평가','모델','128×128 입력의 ResNet-20 평가'],
 ['e13_vela','E13','Arm Vela 비용 대조','비용·외부 검증','두 비용 추정기의 결과 비교; 실리콘 측정 아님'],
 ['e14_rounding_seeds','E14','반올림 반복 실험','정수 출력','3개 체크포인트의 반올림 모드별 정확도'],
 ['e15_fidelity','E15','모의·정수 실행 일치','정수 출력','전체 시험셋의 정확도 및 출력 정숫값 비교'],
 ['e16_ln_emulation','E16','LayerNorm 산술 교체','정수 출력','동일 정수 그래프에서 모의 실행 경로만 변경'],
 ['e17_dense_output','E17','초해상도 출력 비교','정수 출력','ESPCN ×2의 PSNR 및 출력 정숫값 비교']
];
const PRESET_LABELS={
 'tiny-1tops':'tiny-1tops', 'edge-10tops':'edge-10tops',
 'edge-10tops-strict':'edge-10tops-strict','pcie-80tops':'pcie-80tops'
};
function validate(v,key,min,max,integer=false){
 if(typeof v!=='number'||!Number.isFinite(v)||v<min||v>max||(integer&&!Number.isInteger(v)))throw new Error('유효하지 않은 설정: '+key);
 return v;
}
function specWithOverrides(base,changes={}){
 if(!base)throw new Error('가상 NPU 설정이 없습니다.');
 const s=JSON.parse(JSON.stringify(base));
 const allowed=['array','cores','dram','depthwise','activation'];
 if(Object.keys(changes).some(k=>!allowed.includes(k)))throw new Error('지원하지 않는 비용 설정입니다.');
 if('array' in changes){const a=validate(changes.array,'배열 크기',8,256,true);if(![8,16,32,64,128,256].includes(a))throw new Error('지원하지 않는 배열 크기');s.pe_rows=a;s.pe_cols=a;}
 if('cores' in changes)s.cores=validate(changes.cores,'코어 수',1,8,true);
 if('dram' in changes)s.dram_gbps=validate(changes.dram,'DRAM 대역폭',.5,80);
 if('depthwise' in changes){if(typeof changes.depthwise!=='boolean')throw new Error('깊이별 엔진 설정 오류');s.dw_lanes=changes.depthwise?(base.dw_lanes||16):0;}
 if('activation' in changes){
  if(!['lut','fallback'].includes(changes.activation))throw new Error('활성함수 설정 오류');
  const acts=['silu','gelu','hswish'];
  const baseMode=base.lut_acts.includes('gelu')?'lut':'fallback';
  if(changes.activation!==baseMode){
   s.unsupported_ops=s.unsupported_ops.filter(x=>!acts.includes(x));
   s.lut_acts=s.lut_acts.filter(x=>!acts.includes(x));
   if(changes.activation==='lut')s.lut_acts.push(...acts);
   else s.unsupported_ops.push(...acts);
  }
 }
 return s;
}
function unit(row){return row.bound==='fallback'?'host':row.onArray?'array':row.kind==='dwconv'?'depthwise':row.cycles>0?'vector':'structural';}
const UNIT_LABELS={host:'호스트 처리',array:'MAC 배열',depthwise:'Depthwise 엔진',vector:'벡터 연산',structural:'구조·융합 연산'};
function analysis(data,model,preset,overrides={}){
 if(!data.models||!Object.prototype.hasOwnProperty.call(data.models,model))throw new Error('모델 구조 기록이 없습니다.');
 if(!data.presets||!Object.prototype.hasOwnProperty.call(data.presets,preset))throw new Error('가상 NPU 설정이 없습니다.');
 const spec=specWithOverrides(data.presets[preset],overrides);
 const est=C.estimate(data.models[model],spec);
 const custom=JSON.stringify(spec)!==JSON.stringify(data.presets[preset]);
 const groups=new Map(),units=new Map();
 const layers=new Map(data.models[model].layers.map(l=>[l.name,l]));
 for(const row of est.rows){
  const u=unit(row), l=layers.get(row.name);
  const op=row.op==='act'?(l.kind||'activation'):row.kind==='dwconv'?'depthwise conv':row.op;
  const k=op+'|'+u;
  if(!groups.has(k))groups.set(k,{op,unit:u,host:u==='host',count:0,cycles:0});
  const g=groups.get(k);g.count++;g.cycles+=row.cycles;
  if(!units.has(u))units.set(u,{key:u,label:UNIT_LABELS[u],cycles:0,count:0});
  units.get(u).cycles+=row.cycles;units.get(u).count++;
 }
 const operators=[...groups.values()].sort((a,b)=>Number(b.host)-Number(a.host)||b.cycles-a.cycles);
 const hostCycles=units.get('host')?.cycles||0;
 let saved=null;
 if(!custom&&['edge-10tops','edge-10tops-strict'].includes(preset)){
  const x=A.getCase(data,model,preset);
  if(x.after)saved=x;
 }
 return {model,label:A.MODEL_LABELS[model]||model,preset,spec,estimate:est,operators,
  units:[...units.values()].sort((a,b)=>b.cycles-a.cycles),hostCycles,
  hostShare:est.total?hostCycles/est.total:0,hostCount:operators.filter(x=>x.host).reduce((a,b)=>a+b.count,0),
  supported:hostCycles===0,custom,provenance:'simulated',recordedComparison:saved,
  params:data.models[model].total_params,config:data.models[model].config};
}
function research(data,model,scheme){
 const summary=A.getResearch(data,model,scheme);
 const record=A.rows(data,'e15_fidelity').find(r=>r.model===model&&r.scheme===scheme);
 return {summary,localImages:record.agreement_batch.images,layers:record.per_layer_agreement,
  layernorm:model==='cust_vit'?A.getLayerNorm(data,scheme):null,record};
}
function catalog(data){return EXPERIMENTS.map(([key,id,title,group,description])=>({key,id,title,group,description,count:A.rows(data,key).length}));}
function recordValue(data,experiment,index,key){
 const records=A.rows(data,experiment);
 if(!Number.isInteger(index)||index<0||index>=records.length)throw new Error('실험 기록을 찾을 수 없습니다.');
 const r=records[index];
 if(experiment==='e4_surgery'){
  if(key==='condition')return r.variant??r.step??(r.part==='c'&&Number.isFinite(r.qat_epochs)?'QAT · '+r.qat_epochs+' epoch':null);
  if(key==='original_fp32')return r.float_acc??null;
  if(key==='modified_fp32')return r.float_acc_after_heal??r.float_acc_after_cle??null;
  if(key==='simulated_after')return r.fake_acc??r.qat_acc??null;
 }
 if(experiment==='e12_imagenette'&&key==='fp32')return r.test_acc??r.float_acc??null;
 return key.split('.').reduce((v,k)=>v?.[k],r)??null;
}
const api=Object.freeze({analysis,research,catalog,recordValue,specWithOverrides,UNIT_LABELS,PRESET_LABELS,EXPERIMENTS});
if(node)module.exports=api;else root.NpuWorkbench=api;
})(typeof globalThis!=='undefined'?globalThis:this);
