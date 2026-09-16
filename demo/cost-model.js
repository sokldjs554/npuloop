/* Analytical cost model; unchanged numerical implementation from the integrated demo. */
function gemmCycles(m,k,n,spec){ const R=spec.pe_rows,C=spec.pe_cols,fd=spec.fill_drain?(2*R+C-2):0; const kt=Math.ceil(k/R);
  const nPer=Math.ceil(n/spec.cores), cycN=kt*Math.ceil(nPer/C)*(m+fd); const mPer=Math.ceil(m/spec.cores), cycM=kt*Math.ceil(n/C)*(mPer+fd);
  if(spec.cores===1||cycN<=cycM) return [cycN, spec.cores>1?'N':'-']; return [cycM,'M']; }
function estimate(model, spec){ const sram=spec.sram_kb*1024, bpc=spec.dram_gbps*1e9/(spec.freq_mhz*1e6); const resident={}; const outB={}; model.layers.forEach(l=>outB[l.name]=l.elements);
  const inTraffic = l => l.inputs.reduce((b,s)=> b + (resident[s]?0:outB[s]), 0);
  const place=(l,extra=0)=>{ const fits=(outB[l.name]+extra)<=sram; resident[l.name]=fits; return fits?0:outB[l.name]; };
  const rows=[]; const macsPerCycle=spec.pe_rows*spec.pe_cols*spec.cores; let total=0, totalMacs=0, arrayMacs=0, dram=0;
  for(const l of model.layers){ const r={name:l.name,op:l.op,kind:l.op,macs:l.macs||0,cycles:0,compute:0,dramB:0,dramCyc:0,bound:'none',util:0,engineUtil:0,onArray:false,split:'',m:l.m||0,k:l.k||0,n:l.n||0};
    if(l.op==='input'){resident[l.name]=false; rows.push(r); continue;}
    if(['output','flatten','reshape'].includes(l.op)){resident[l.name]=resident[l.inputs[0]]||false; rows.push(r); continue;}
    if(l.op==='conv'||l.op==='linear'){ const liveIn=l.in_bytes.reduce((a,b)=>a+b,0);
      if(l.op==='conv'&&l.depthwise){ r.kind='dwconv'; if(spec.dw_lanes>0){ const chPer=Math.ceil(l.n/spec.cores); r.compute=Math.ceil(chPer/spec.dw_lanes)*l.kh*l.kw*l.m; r.split='C'; }
        else { const one={...spec,cores:1}; const [per]=gemmCycles(l.m,l.kh*l.kw,1,one); r.compute=Math.ceil(l.n/spec.cores)*per; r.onArray=true; r.split='C'; } }
      else if(l.groups>1){ r.kind='gconv'; const [per,sp]=gemmCycles(l.m,l.k,l.n/l.groups,spec); r.compute=per*l.groups; r.split=sp; r.onArray=true; }
      else { r.kind=l.op; const [c,sp]=gemmCycles(l.m,l.k,l.n,spec); r.compute=c; r.split=sp; r.onArray=true; }
      r.dramB=l.weight_bytes+inTraffic(l)+place(l,liveIn); r.dramCyc=r.dramB/bpc; r.cycles=Math.max(r.compute,r.dramCyc); r.bound=r.compute>=r.dramCyc?'compute':'memory';
      if(r.onArray) r.util=l.macs/(r.cycles*macsPerCycle); else if(r.kind==='dwconv') r.engineUtil=l.macs/(r.cycles*spec.dw_lanes*spec.cores); }
    else if(l.op==='act'){ if(spec.fused_acts.includes(l.kind)){ r.kind='act-fused'; r.cycles=0; resident[l.name]=resident[l.inputs[0]]||false; }
      else if(spec.unsupported_ops.includes(l.kind)||!spec.lut_acts.includes(l.kind)){ r.kind='act-fallback'; r.dramB=l.elements; r.dramCyc=r.dramB/bpc; r.cycles=l.elements*spec.fallback_cycles_per_elem+r.dramCyc; r.bound='fallback'; resident[l.name]=false; }
      else { r.kind='act-lut'; const v=Math.ceil(l.elements/(spec.vector_lanes*spec.cores)); r.dramB=inTraffic(l)+place(l); r.dramCyc=r.dramB/bpc; r.cycles=Math.max(v,r.dramCyc); r.bound=v>=r.dramCyc?'vector':'memory'; } }
    else if(l.op==='const'){ resident[l.name]=false; rows.push(r); continue; }
    else if(l.op==='matmul'){
      const [per,sp]=gemmCycles(l.m,l.k,l.n,spec); r.compute=per*l.batch; r.split=sp; r.onArray=true;
      r.dramB=inTraffic(l)+place(l); r.dramCyc=r.dramB/bpc; r.cycles=Math.max(r.compute,r.dramCyc);
      r.bound=r.compute>=r.dramCyc?'compute':'memory'; r.util=r.cycles?l.macs/(r.cycles*macsPerCycle):0;
    }
    else if(l.op==='softmax'||l.op==='layernorm'){
      const elems=l.in_bytes[0], passes=l.op==='softmax'?spec.softmax_passes:spec.layernorm_passes;
      if(spec.unsupported_ops.includes(l.op)){
        r.kind=l.op+'-fallback';r.dramB=elems;r.dramCyc=r.dramB/bpc;
        r.cycles=elems*spec.fallback_cycles_per_elem+r.dramCyc;r.bound='fallback';resident[l.name]=false;
      }else{
        const v=passes*Math.ceil(elems/(spec.vector_lanes*spec.cores));
        r.dramB=inTraffic(l)+place(l);r.dramCyc=r.dramB/bpc;r.cycles=Math.max(v,r.dramCyc);r.bound=v>=r.dramCyc?'vector':'memory';
      }
    }
    else if(l.op==='transpose'||l.op==='mul'){
      if(l.op==='mul'&&l.kind==='scalar'){r.kind='mul-fused';resident[l.name]=resident[l.inputs[0]]||false;}
      else{
        const elems=l.op==='transpose'?l.in_bytes[0]:l.elements, v=Math.ceil(elems/(spec.vector_lanes*spec.cores));
        r.dramB=inTraffic(l)+place(l);r.dramCyc=r.dramB/bpc;r.cycles=Math.max(v,r.dramCyc);r.bound=v>=r.dramCyc?'vector':'memory';
      }
    }
    else if(['add','pool','concat'].includes(l.op)){ r.kind=l.op; const elems=l.op==='pool'?l.in_bytes[0]:l.in_bytes.reduce((a,b)=>a+b,0); const v=Math.ceil(elems/(spec.vector_lanes*spec.cores)); r.dramB=inTraffic(l)+place(l); r.dramCyc=r.dramB/bpc; r.cycles=Math.max(v,r.dramCyc); r.bound=v>=r.dramCyc?'vector':'memory'; }
    else { throw new Error('cost model: unhandled op '+l.op); }
    total+=r.cycles; totalMacs+=r.macs; if(r.onArray) arrayMacs+=r.macs; dram+=r.dramB; rows.push(r); }
  return {rows,total,totalMacs,dram,ideal:totalMacs/macsPerCycle,util:total?arrayMacs/(total*macsPerCycle):0,latency_ms:total/(spec.freq_mhz*1e3),peak_tops:2*macsPerCycle*spec.freq_mhz*1e6/1e12}; }


if(typeof module!=="undefined" && module.exports) module.exports={gemmCycles,estimate};
