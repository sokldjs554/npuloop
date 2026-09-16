/* Pure adapters for recorded evidence. UI state never changes these source records. */
(function (root) {
  'use strict';
  const MODEL_LABELS = Object.freeze({
    cust_vit: 'ViT-128/6', cust_inception: 'Inception-32',
    resnet20_relu: 'ResNet-20 · ReLU', resnet20_silu: 'ResNet-20 · SiLU',
    mnv2_050_relu6: 'MobileNetV2-0.5', imagenette_resnet20: 'ResNet-20 · Imagenette'
  });
  const PRESET_LABELS = Object.freeze({
    'edge-10tops-strict': '지원 제한형 · strict',
    'edge-10tops': 'LUT·정규화 지원형'
  });
  function rows(data, source) {
    const r = data && data.results && data.results[source];
    return r && Array.isArray(r.records) ? r.records : [];
  }
  function number(value, name, min, max) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
      throw new Error('저장된 수치를 확인할 수 없습니다: ' + name);
    }
    return value;
  }
  const nonnegative = (v, name) => number(v, name, 0, Number.MAX_VALUE);
  const rate = (v, name) => number(v, name, 0, 1);
  const optionalRate = (v, name) => v == null ? null : rate(v, name);
  function getCase(data, model, preset) {
    if (!Object.prototype.hasOwnProperty.call(PRESET_LABELS, preset)) {
      throw new Error('저장된 사례가 없는 가상 NPU 조건입니다.');
    }
    const r = rows(data, 'e9_customer_intake').find(x => x.model === model);
    if (!r) throw new Error('저장된 E9 모델 사례가 없습니다.');
    const key = preset === 'edge-10tops-strict' ? 'intake_strict' : 'intake';
    const report = r[key];
    if (!report || !report.diagnose || !report.receive || !report.model) {
      throw new Error('모델의 접수·진단 기록이 불완전합니다.');
    }
    const cycles = number(report.diagnose.cycles, 'cycles', Number.MIN_VALUE, Number.MAX_VALUE);
    const hostOps = report.receive.ops.filter(x => x.unit === 'host').map(x => ({op:x.op,count:x.count}));
    const units = report.diagnose.by_unit;
    if (!units || (hostOps.length && !Object.prototype.hasOwnProperty.call(units, 'host'))) {
      throw new Error('호스트 비용 분해 기록이 없습니다.');
    }
    const hostCycles = Object.prototype.hasOwnProperty.call(units, 'host')
      ? number(units.host, 'host cycles', 0, cycles) : 0;
    const ptq = r.ptq || {};
    const result = {
      source:'e9_customer_intake', model, label:MODEL_LABELS[model] || model,
      preset, presetLabel:PRESET_LABELS[preset],
      config:{...report.model.config}, params:r.params, macs:r.macs,
      cycles, hostCycles, hostShare:hostCycles/cycles, hostOps,
      fullySupported:report.receive.runs_on_chip === true,
      floatAccuracy:optionalRate(r.float_acc, 'float_acc'),
      intAccuracy:optionalRate(ptq.int_acc, 'int_acc'),
      intImages:ptq.int_eval_images == null ? null : nonnegative(ptq.int_eval_images, 'int_eval_images'),
      after:null, recordedAt:r.timestamp || null,
      alternatives:(report.alternatives || []).map(x => ({...x})),
      report
    };
    if (r.after) {
      const a = r.after;
      const afterCycles = nonnegative(preset === 'edge-10tops-strict' ? a.cycles_strict : a.cycles, 'after cycles');
      // Only this explicitly recorded surgery has a supported operation-list interpretation.
      // The remaining support status is an inference from the operation list, not a hardware run.
      const supportKnown = model === 'cust_vit' && a.action === "swap {'gelu': 'relu'} + heal 3 epochs";
      const remaining = supportKnown ? hostOps.filter(x => x.op !== 'act:gelu') : null;
      result.after = {
        action:a.action, cycles:afterCycles, saving:1-afterCycles/cycles,
        floatAccuracy:optionalRate(a.float_acc, 'after float_acc'),
        intAccuracy:optionalRate(a.int_acc, 'after int_acc'),
        intImages:a.int_eval_images == null ? null : nonnegative(a.int_eval_images,'after int_eval_images'),
        remainingHostOps:remaining, supportInferred:supportKnown,
        fullySupported:supportKnown ? remaining.length === 0 : null
      };
    }
    return result;
  }
  function getResearch(data, model, scheme) {
    const r = rows(data,'e15_fidelity').find(x => x.model === model && x.scheme === scheme);
    if (!r) throw new Error('선택한 모델·양자화 방식의 E15 결과가 없습니다.');
    const stats = r.int_vs_fake;
    if (!stats || !Array.isArray(stats.ci95) || stats.ci95.length !== 2) {
      throw new Error('대응 비교의 신뢰구간 기록이 없습니다.');
    }
    return {
      source:'e15_fidelity', model, label:MODEL_LABELS[model] || model,
      scheme, dataset:r.dataset, images:nonnegative(r.n_test,'n_test'),
      fakeAccuracy:rate(r.fake_acc,'fake_acc'), intAccuracy:rate(r.int_acc,'int_acc'),
      delta:number(stats.delta,'delta',-1,1), se:nonnegative(stats.se,'se'),
      ci95:stats.ci95.map(v => number(v,'ci95',-1,1)),
      mismatch:rate(r.output_codes.mismatch_frac,'mismatch_frac'),
      labelAgreement:rate(stats.top1_agreement,'top1_agreement')
    };
  }
  function getLayerNorm(data, scheme) {
    const rr=rows(data,'e16_ln_emulation');
    const b=rr.find(x => x.variant === 'float-ln' && x.scheme === scheme);
    const a=rr.find(x => x.variant === 'int-ln' && x.scheme === scheme);
    if (!b || !a) return null;
    if (b.n_test !== a.n_test || b.agreement_batch.images !== a.agreement_batch.images) {
      throw new Error('E16 비교 조건의 표본 수가 다릅니다.');
    }
    return {
      source:'e16_ln_emulation', scheme,
      localImages:nonnegative(b.agreement_batch.images,'local images'),
      outputImages:nonnegative(b.n_test,'output images'),
      beforeLocal:rate(b.per_op.layernorm.local_mean,'before local'),
      afterLocal:rate(a.per_op.layernorm.local_mean,'after local'),
      beforeMismatch:rate(b.output_codes.mismatch_frac,'before mismatch'),
      afterMismatch:rate(a.output_codes.mismatch_frac,'after mismatch')
    };
  }
  const api = Object.freeze({MODEL_LABELS,PRESET_LABELS,rows,getCase,getResearch,getLayerNorm});
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.NpuDemoData = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
