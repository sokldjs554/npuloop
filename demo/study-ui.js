/* Model research workspace: recorded evidence and explicitly imported local runs. */
(function(root){
'use strict';
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=(x,d=0)=>finite(x)?x.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d}):'미기록';
const pct=x=>finite(x)?num(x*100,2)+'%':'미기록';
const compact=x=>!finite(x)?'미기록':x>=1e6?num(x/1e6,2)+'M':x>=1e3?num(x/1e3,1)+'K':num(x);
const STAGES=['baseline_fp32','changed_fp32_before_training','fine_tuned_fp32','fake_quant','integer'];
const STAGE_LABELS=['기준 FP32','수정 직후 FP32','회복 학습 후 FP32','모의 양자화','정수 엔진'];

function validateReport(report){
 if(!report||report.schema_version!=='npuloop.model-study.v1')throw new Error('지원하지 않는 실행 결과 형식입니다. STUDY_RUNNER.md의 JSON 형식을 사용하세요.');
 if(report.status!=='complete')throw new Error('완료된 실행 결과만 가져올 수 있습니다.');
 const observations=STAGES.map(key=>report.observed?.[key]);
 for(const row of observations){
  if(!row||row.provenance!=='measured_cpu'||!finite(row.accuracy)||row.accuracy<0||row.accuracy>1||!Number.isInteger(row.n)||row.n<=0||!Number.isInteger(row.correct)||row.correct<0||row.correct>row.n||Math.abs(row.accuracy-row.correct/row.n)>1e-8||row.split!=='test'||!/^[a-f0-9]{64}$/.test(row.subset_sha256||'')||!/^[a-f0-9]{64}$/.test(row.model_state_sha256||''))throw new Error('평가 정확도, 이미지 수 또는 출처 해시가 올바르지 않습니다.');
 }
 if(observations.some(row=>row.n!==observations[0].n||row.subset_sha256!==observations[0].subset_sha256))throw new Error('평가 단계의 시험 부분집합이 다릅니다. 같은 이미지로 다시 실행하세요.');
 if(report.simulated_costs?.provenance!=='simulated'||!['baseline','changed'].every(key=>finite(report.simulated_costs?.[key]?.total_cycles)&&report.simulated_costs[key].total_cycles>0))throw new Error('가상 NPU 비용의 출처 또는 값이 올바르지 않습니다.');
 if(!Array.isArray(report.training?.epochs)||report.training.epochs.length>10000||(!report.training.epochs.length&&!(report.config?.epochs===0&&report.training.actual_steps===0)))throw new Error('실행 결과에 학습 epoch 기록이 없습니다.');
 for(const row of report.training.epochs){
  if(!Number.isInteger(row.epoch)||row.epoch<1||!finite(row.train_loss)||row.train_loss<0||!finite(row.val_acc)||row.val_acc<0||row.val_acc>1)throw new Error('학습 기록의 epoch, loss 또는 검증 정확도가 올바르지 않습니다.');
 }
 return report;
}

function mount(data,actions){
 const $=id=>document.getElementById(id), S=root.NpuStudy;
 const state={id:'e4-activation',preset:'edge-10tops-strict',variant:'swap-relu-heal3',task:'all',maxDropPp:1,minSavingPercent:0};
 const kinds={activation:'활성함수 수정·회복 학습',pruning:'채널 프루닝·회복 학습',training:'기준 모델 학습',quantization:'양자화 실험',qat:'양자화 인지 학습',dataset:'데이터·입력 크기 적응'};
 let current,selected;
 const fill=(el,items,value)=>{el.innerHTML=items.map(([v,t])=>`<option value="${esc(v)}">${esc(t)}</option>`).join('');el.value=value;};
 const label=v=>({'silu-ptq':'SiLU · PTQ 기준','silu-qat':'SiLU · QAT','swap-relu-heal0':'ReLU 교체 · 학습 전','swap-relu-heal3':'ReLU 교체 · 3 epoch 회복','swap-relu-heal1':'ReLU 교체 · 1 epoch 회복'})[v.id]||v.label;
 const card=(title,value,note,cls='')=>`<div class="metric ${cls}"><div class="metric-label">${esc(title)}</div><div class="metric-value">${esc(value)}</div><div class="metric-note">${esc(note)}</div></div>`;
 const sample=(v,key)=>finite(v[key])?num(v[key])+'장':'평가 수 미기록';
 function selection(){
  selected=current.variants.find(v=>v.id===state.variant);
  if(!selected){selected=current.variants.find(v=>v.id==='swap-relu-heal3')||current.variants.find(v=>v.id==='after')||current.variants.find(v=>v.strategy==='uniform'&&v.ratio===0.5)||current.variants.find(v=>v.id===current.baselineId)||current.variants[0];state.variant=selected.id;}
 }
 function render(id){
  const all=S.catalog(data);
  if(id&&all.some(s=>s.id===id)&&id!==state.id){state.id=id;state.variant=null;state.task='all';$('study-task').value='all';}
  let items=all.filter(s=>state.task==='all'||s.task===state.task);
  if(!items.some(s=>s.id===state.id)){state.id=items[0].id;state.variant=null;}
  fill($('study-select'),items.map(s=>[s.id,s.title]),state.id);
  fill($('study-preset'),['edge-10tops-strict','edge-10tops','tiny-1tops','pcie-80tops'].filter(k=>data.presets[k]).map(k=>[k,k]),state.preset);
  current=S.study(data,state.id,state.preset);selection();
  $('study-kind').textContent=(kinds[current.kind]||current.kind)+' · '+current.sourceKey.split('_')[0].toUpperCase();
  $('study-title').textContent=current.title;
  $('study-subtitle').textContent=(current.task==='classification'?'이미지 분류':'초해상도')+' / '+current.dataset+' / 저장된 실험 결과';
  renderSelected();renderVariants();renderDecision();
 }
 function renderSelected(){
  const v=selected, base=current.variants.find(x=>x.id===current.baselineId), sr=current.task==='super-resolution';
  const same=current.variants.length>1&&v.id!==base.id;
  const immediate=v.afterSwapAccuracy??v.afterPruneAccuracy;
  const stages=[['기준 모델',sr?num(base.fp32Metric,2)+' dB':pct(base.fp32Accuracy)],['구조·방식 변경',finite(immediate)?pct(immediate):label(v)],['학습',finite(v.epochs)?num(v.epochs)+' epoch':'로그 미기록'],['양자화',sr?num(v.fakeMetric,2)+' dB':pct(v.fakeAccuracy)],['정수 평가',sr?num(v.intMetric,2)+' dB':pct(v.intAccuracy)]];
  $('study-context').innerHTML=`<div class="study-pipeline">${stages.map(([title,value],i)=>`<div><span class="step-index">0${i+1}</span><span>${esc(title)}</span><strong>${esc(value)}</strong></div>`).join('')}</div><p class="footnote study-scope">${sr?'PSNR · '+sample(v,'metricImages'):'FP32 '+sample(v,'fp32Images')+' · 모의 양자화 '+sample(v,'fakeImages')+' · 정수 엔진 '+sample(v,'intImages')}<span>선택: ${esc(label(v))}</span></p>`;
  $('study-metrics').innerHTML=sr?
   card('FP32 · PSNR',finite(v.fp32Metric)?num(v.fp32Metric,2)+' dB':'미기록',sample(v,'metricImages'))+card('모의 양자화 · PSNR',finite(v.fakeMetric)?num(v.fakeMetric,2)+' dB':'미기록','이미지별 PSNR 평균')+card('정수 엔진 · PSNR',finite(v.intMetric)?num(v.intMetric,2)+' dB':'미기록','분류 정확도와 별도 지표')+card('정수 경로 PSNR 변화',finite(v.intMetric)?num(v.intMetric-v.fp32Metric,2)+' dB':'미기록','같은 시험 이미지 기준'):
   card('선택 모델 · FP32',pct(v.fp32Accuracy),same&&finite(base.fp32Accuracy)&&finite(v.fp32Accuracy)?'기준 대비 '+num((v.fp32Accuracy-base.fp32Accuracy)*100,2)+' %p':sample(v,'fp32Images'))+card('모의 양자화',pct(v.fakeAccuracy),sample(v,'fakeImages'))+card('정수 엔진',pct(v.intAccuracy),sample(v,'intImages'))+card('추정 사이클',compact(v.cycles),state.preset);
  $('study-training-caption').textContent=label(v)+' · 검증 정확도';
  $('study-training').innerHTML=trainingChart(v.training);
  $('study-tradeoff').innerHTML=tradeoffChart();
  $('study-metric-notes').textContent=[...current.notes,...current.metricNotes].join(' ');
 }
 function trainingChart(rows){
  const valid=(rows||[]).filter(r=>finite(r.epoch)&&finite(r.valAccuracy));
  if(!valid.length)return '<div class="study-empty"><span class="empty-symbol">—</span><p>epoch별 학습 기록 없음</p><small>저장된 최종 평가값은 아래 비교표에 표시됩니다.</small></div>';
  const width=480,height=230,left=48,right=24,top=22,bottom=40;
  const minE=Math.min(...valid.map(r=>r.epoch)),maxE=Math.max(...valid.map(r=>r.epoch));
  const min=Math.max(0,Math.floor((Math.min(...valid.map(r=>r.valAccuracy))*100-1)/5)*5),max=Math.min(100,Math.ceil((Math.max(...valid.map(r=>r.valAccuracy))*100+1)/5)*5);
  const x=e=>left+(maxE===minE?0.5:(e-minE)/(maxE-minE))*(width-left-right),y=a=>top+(max-a*100)/(max-min||1)*(height-top-bottom);
  const points=valid.map(r=>`${x(r.epoch)},${y(r.valAccuracy)}`).join(' ');
  const last=valid[valid.length-1];
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="epoch별 검증 정확도: ${valid.map(r=>'epoch '+r.epoch+' '+pct(r.valAccuracy)).join(', ')}">${[0,1,2].map(i=>{const a=min+(max-min)*i/2,yy=y(a/100);return `<line class="gridline" x1="${left}" x2="${width-right}" y1="${yy}" y2="${yy}"/><text x="${left-8}" y="${yy+4}" text-anchor="end">${num(a,0)}%</text>`;}).join('')}<polyline class="study-curve" points="${points}"/>${valid.map(r=>`<circle cx="${x(r.epoch)}" cy="${y(r.valAccuracy)}" r="${valid.length>20?2:4}" class="study-point"><title>Epoch ${r.epoch} · 검증 ${pct(r.valAccuracy)} · 훈련 loss ${num(r.trainLoss,4)}</title></circle>`).join('')}${[...new Set([minE,Math.round((minE+maxE)/2),maxE])].map(e=>`<text x="${x(e)}" y="${height-19}" text-anchor="middle">${e}</text>`).join('')}<text x="${width-right}" y="${height-3}" text-anchor="end">epoch</text></svg><div class="study-chart-foot"><span>검증 정확도 <b>${pct(last.valAccuracy)}</b></span><span>마지막 train loss <b>${num(last.trainLoss,4)}</b></span></div>`;
 }
 function tradeoffChart(){
  const rows=current.variants.filter(v=>finite(v.fp32Accuracy)&&finite(v.cycles));
  if(rows.length<2)return '<div class="study-empty"><span class="empty-symbol">—</span><p>정확도·비용의 공통 기록 부족</p><small>기록되지 않은 비용이나 정확도를 추정하지 않습니다.</small></div>';
  const minC=Math.min(...rows.map(v=>v.cycles)),maxC=Math.max(...rows.map(v=>v.cycles));
  const minA=Math.max(0,Math.floor(Math.min(...rows.map(v=>v.fp32Accuracy))*100-1)),maxA=Math.min(100,Math.ceil(Math.max(...rows.map(v=>v.fp32Accuracy))*100+1));
  const x=c=>55+(maxC===minC?0.5:(c-minC)/(maxC-minC))*385,y=a=>28+(maxA-a*100)/(maxA-minA||1)*145;
  return `<svg viewBox="0 0 480 230" role="img" aria-label="FP32 정확도와 추정 사이클 비교">${[0,1,2].map(i=>{const a=minA+(maxA-minA)*i/2,yy=y(a/100);return `<line class="gridline" x1="55" x2="450" y1="${yy}" y2="${yy}"/><text x="47" y="${yy+4}" text-anchor="end">${num(a,1)}%</text>`;}).join('')}${rows.map(v=>`<circle class="study-scatter ${v.id===state.variant?'chosen':''}" cx="${x(v.cycles)}" cy="${y(v.fp32Accuracy)}" r="${v.id===state.variant?7:5}"><title>${esc(label(v))}: ${pct(v.fp32Accuracy)}, ${num(v.cycles)} cycles</title></circle>`).join('')}<text x="55" y="201" text-anchor="start">${compact(minC)}</text><text x="440" y="201" text-anchor="end">${compact(maxC)}</text><text x="440" y="222" text-anchor="end">추정 사이클 →</text></svg><div class="study-chart-foot"><span><i class="dot blue"></i> 선택 모델</span><span>높은 정확도 · 적은 사이클</span></div>`;
 }
 function renderVariants(){
  const sr=current.task==='super-resolution';
  $('study-variant-count').textContent=current.variants.length+'개 변경안';
  const headings=sr?['변경안','FP32 PSNR','모의 PSNR','정수 PSNR','평가 이미지']:['변경안','FP32','모의 양자화','정수 엔진','파라미터','MACs','추정 사이클'];
  $('study-variants').innerHTML=`<table><thead><tr>${headings.map((h,i)=>`<th${i?' class="num"':''}>${h}</th>`).join('')}</tr></thead><tbody>${current.variants.map(v=>`<tr class="${v.id===state.variant?'selected':''}"><td><button class="study-row-button" data-study-variant="${esc(v.id)}" aria-pressed="${v.id===state.variant}">${esc(label(v))}</button><span class="study-row-note">${v.id===current.baselineId?'비교 기준':finite(v.epochs)?v.epochs+' epoch':'학습 횟수 미기록'}</span></td>${(sr?[finite(v.fp32Metric)?num(v.fp32Metric,3)+' dB':'미기록',finite(v.fakeMetric)?num(v.fakeMetric,3)+' dB':'미기록',finite(v.intMetric)?num(v.intMetric,3)+' dB':'미기록',num(v.metricImages)]:[pct(v.fp32Accuracy),pct(v.fakeAccuracy),pct(v.intAccuracy),compact(v.params),compact(v.macs),num(v.cycles)]).map(t=>`<td class="num${t==='미기록'?' muted':''}">${esc(t)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
 }
 function renderDecision(){
  const sr=current.task!=='classification';$('study-budget-form').hidden=sr;
  if(sr){$('study-decision').innerHTML='<p>초해상도는 PSNR로 비교합니다. 분류 정확도 %p 조건을 적용하지 않습니다.</p>';return;}
  const assessment=S.assess(current,state),changed=assessment.filter(a=>a.variantId!==current.baselineId),eligible=changed.filter(a=>a.eligible),unknown=changed.filter(a=>a.eligible===null);
  $('study-decision').innerHTML=eligible.length?`<strong class="blue-text">조건 충족 ${eligible.length}개</strong><div class="study-candidates">${eligible.map(a=>`<button class="btn subtle" data-study-variant="${esc(a.variantId)}">${esc(label(current.variants.find(v=>v.id===a.variantId)))}</button>`).join('')}</div><p>저장된 단일 실행 결과 기준입니다. 선택 조건 통과가 실측 NPU 성능이나 반복 실험의 통계적 유의성을 뜻하지 않습니다.</p>`:`<strong>${unknown.length===changed.length?'판정에 필요한 기록 부족':'조건을 만족하는 변경 없음'}</strong><p>${unknown.length?unknown.length+'개 변경안은 같은 평가 범위의 FP32 또는 비용이 없어 판정을 보류했습니다.':'기준 모델을 제외한 모든 변경안이 현재 조건을 충족하지 않습니다.'}</p>`;
 }
 $('study-select').addEventListener('change',()=>{state.id=$('study-select').value;state.variant=null;render();history.replaceState(null,'','#studies/'+state.id);});
 $('study-task').addEventListener('change',()=>{state.task=$('study-task').value;render();history.replaceState(null,'','#studies/'+state.id);});
 $('study-preset').addEventListener('change',()=>{state.preset=$('study-preset').value;render();});
 $('study-budget-form').addEventListener('submit',e=>{e.preventDefault();state.maxDropPp=Number($('study-max-drop').value);state.minSavingPercent=Number($('study-min-saving').value);renderDecision();});
 $('view-studies').addEventListener('click',e=>{const button=e.target.closest('[data-study-variant]');if(button){const restore=$('study-variants').contains(button);state.variant=button.dataset.studyVariant;selection();renderSelected();renderVariants();if(restore)[...$('study-variants').querySelectorAll('[data-study-variant]')].find(b=>b.dataset.studyVariant===state.variant)?.focus();}});
 $('study-source').addEventListener('click',()=>actions.showRecord(current.title+' · 원본',data.results[current.sourceKey],current.sourceKey+'.json'));
 $('study-export').addEventListener('click',()=>actions.download('npuloop_'+state.id+'.json',{schema_version:'npuloop.recorded-study.v1',provenance:'stored_experiment_records',hardware_measured:false,study:current,selected_variant:state.variant,constraints:{maxDropPp:state.maxDropPp,minSavingPercent:state.minSavingPercent},assessment:S.assess(current,state),source:data.results[current.sourceKey]}));
 $('study-import').addEventListener('change',async()=>{
  $('study-import-status').textContent='';$('study-import-result').innerHTML='';
  try{
   const file=$('study-import').files[0];if(!file)return;if(file.size>5*1024*1024)throw new Error('결과 JSON은 5 MB 이하만 가져올 수 있습니다.');
   const r=validateReport(JSON.parse(await file.text()));
   const observations=STAGES.map(k=>r.observed[k]);
   $('study-import-status').textContent=file.name+' · 형식과 평가 범위 확인';
   $('study-import-result').innerHTML=`<div class="study-import-heading"><h3>사용자 실행 결과</h3><span class="badge neutral">가져온 파일 · 외부 인증 없음</span></div><p class="footnote">동일 시험 부분집합 ${num(observations[0].n)}장 · 학습 ${r.training.epochs.length} epoch · NPU 비용은 추정값</p><div class="table-container"><table><thead><tr><th>단계</th><th class="num">정확도</th><th class="num">정답 / 이미지</th></tr></thead><tbody>${observations.map((o,i)=>`<tr><td>${STAGE_LABELS[i]}</td><td class="num">${pct(o.accuracy)}</td><td class="num">${num(o.correct)} / ${num(o.n)}</td></tr>`).join('')}</tbody></table></div><div class="study-local-chart">${trainingChart(r.training.epochs.map(e=>({epoch:e.epoch,trainLoss:e.train_loss,valAccuracy:e.val_acc})))}</div><p class="footnote">추정 사이클: ${num(r.simulated_costs.baseline.total_cycles)} → ${num(r.simulated_costs.changed.total_cycles)} · ${esc(r.simulated_costs.preset)}<br>시험 부분집합 SHA-256: <span class="study-hash">${esc(observations[0].subset_sha256)}</span></p>`;
  }catch(e){$('study-import-status').textContent=e.message;}
 });
 return{render,getStudy:()=>current};
}
const api={mount,validateReport};if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.NpuStudyUI=api;
})(typeof globalThis!=='undefined'?globalThis:this);
