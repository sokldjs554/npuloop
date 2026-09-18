/* Recorded research studies. This module never estimates accuracy or training curves.
 * Accuracy fields are fractions; classification metric values are percentages.
 * sourceRecords contains JSON pointers into data.results[sourceKey].
 */
(function(root) {
'use strict';
const LABELS = {
  resnet20_relu: 'ResNet-20 · ReLU', resnet20_silu: 'ResNet-20 · SiLU',
  mnv2_050_relu6: 'MobileNetV2 · 0.5', cust_vit: 'ViT-128/6',
  cust_inception: 'Inception-32', espcn_x2: 'ESPCN ×2 · conv 3', espcn_x2_deep: 'ESPCN ×2 · conv 9'
};
const CLASSIFICATION = 'classification';
const own = (object, key) => !!object && Object.prototype.hasOwnProperty.call(object, key);
const number = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
const rows = (data, key) => Array.isArray(data?.results?.[key]?.records) ? data.results[key].records : [];
const meta = (data, key) => data?.results?.[key]?.meta || {};
const copy = value => value == null ? null : JSON.parse(JSON.stringify(value));
const pointer = (sourceKey, recordIndex, field = '') => ({
  sourceKey, recordIndex, pointer: '/records/' + recordIndex + (field ? '/' + field : '')
});
const references = (data, key, predicate = () => true) => rows(data, key)
  .map((record, index) => ({record, index})).filter(({record}) => predicate(record));
const baselineRecord = (data, model) => references(data, 'e1_baselines', r => r.model === model)[0];
const dataset = (data) => meta(data, 'e1_baselines').dataset || 'CIFAR-10';
const testImages = (data, model) => number(baselineRecord(data, model)?.record.splits?.test);
const scope = (name, images) => images === null ? null : name + ':test:all:' + images;
const cost = (record, preset) => number(record?.cost?.[preset]?.total_cycles);

function catalog(data) {
  const found = [];
  function add(id, title, kind, task, ds, model, sourceKey) {
    found.push({id, title, kind, task, dataset: ds, model, sourceKey});
  }
  const cifar = dataset(data);
  if (rows(data, 'e4_surgery').some(r => r.part === 'b'))
    add('e4-activation', 'SiLU 활성함수 교체·회복 학습', 'activation', CLASSIFICATION, cifar, 'resnet20_silu', 'e4_surgery');
  if (rows(data, 'e1_baselines').length)
    add('e1-training', '모델별 기준 학습·규모', 'training', CLASSIFICATION, cifar, 'all', 'e1_baselines');
  for (const model of new Set(rows(data, 'e6_pruning').map(r => r.model).filter(Boolean)))
    add('e6-' + model, (LABELS[model] || model) + ' · 채널 프루닝', 'pruning', CLASSIFICATION, cifar, model, 'e6_pruning');
  if (rows(data, 'e9_customer_intake').some(r => r.model === 'cust_vit'))
    add('e9-vit', 'ViT GELU 교체·회복 학습', 'activation', CLASSIFICATION, cifar, 'cust_vit', 'e9_customer_intake');
  for (const model of new Set(rows(data, 'e2_ptq_grid').map(r => r.model).filter(Boolean)))
    add('e2-' + model, (LABELS[model] || model) + ' · PTQ 방식', 'quantization', CLASSIFICATION, cifar, model, 'e2_ptq_grid');
  for (const model of new Set(rows(data, 'e4_surgery').filter(r => r.part === 'c').map(r => r.model)))
    add('e4-qat-' + model, (LABELS[model] || model) + ' · QAT', 'qat', CLASSIFICATION, cifar, model, 'e4_surgery');
  if (rows(data, 'e4_surgery').some(r => r.part === 'a' && r.model === 'mnv2_050_relu6'))
    add('e4-cle-mnv2_050_relu6', 'MobileNetV2 · CLE·바이어스 보정·QAT', 'quantization', CLASSIFICATION, cifar, 'mnv2_050_relu6', 'e4_surgery');
  if (rows(data, 'e12_imagenette').length)
    add('e12-imagenette', 'Imagenette · 입력 크기·PTQ', 'dataset', CLASSIFICATION,
      meta(data, 'e12_imagenette').dataset || 'Imagenette', 'resnet20_relu_128', 'e12_imagenette');
  // One entry per depth: E17 measures the same width profile at 3 and 9 convolutions, and the point of
  // the experiment is the contrast between them, so they must not collapse into one card.
  for (const model of new Set(rows(data, 'e17_dense_output').map(r => r.model).filter(Boolean)))
    add('e17-' + model, (LABELS[model] || model) + ' · 초해상도 양자화', 'quantization', 'super-resolution',
      meta(data, 'e17_dense_output').dataset || 'Imagenette', model, 'e17_dense_output');
  return found;
}

function training(log) {
  if (!Array.isArray(log)) return [];
  return log.map(row => ({epoch: number(row.epoch), trainLoss: number(row.train_loss),
    trainAccuracy: number(row.train_acc), valAccuracy: number(row.val_acc)}));
}

function variant(id, label, action, fields = {}) {
  return Object.assign({id, label, action, epochs: null, fp32Accuracy: null, fakeAccuracy: null,
    intAccuracy: null, fp32Images: null, fakeImages: null, intImages: null, params: null, macs: null,
    cycles: null, training: [], sourceRecords: [], accuracyScope: null, cycleScope: null,
    fp32OnIntSubset: null, fakeOnIntSubset: null, calibration: null, trainingConfig: null}, fields);
}

function classificationFields(data, model, fp32, fake, integer, intImages) {
  const images = testImages(data, model);
  return {fp32Accuracy: number(fp32), fakeAccuracy: number(fake), intAccuracy: number(integer),
    fp32Images: number(fp32) === null ? null : images,
    fakeImages: number(fake) === null ? null : images,
    intImages: number(integer) === null ? null : number(intImages),
    accuracyScope: number(fp32) === null ? null : scope(dataset(data), images)};
}

function cycleFields(value, sourceKey, preset) {
  const cycles = number(value);
  return {cycles, cycleScope: cycles === null ? null : sourceKey + ':' + preset + ':batch1'};
}

function study(data, id, preset) {
  if (!own(data?.presets, preset)) throw new Error('가상 NPU 설정을 찾을 수 없습니다: ' + preset);
  const descriptor = catalog(data).find(row => row.id === id);
  if (!descriptor) throw new Error('연구 기록을 찾을 수 없습니다: ' + id);
  const result = {...descriptor, preset, baselineId: null, variants: [], notes: [], metricNotes: []};
  const key = descriptor.sourceKey, model = descriptor.model, metadata = meta(data, key);
  const append = (...items) => result.variants.push(...items);

  if (id === 'e4-activation') {
    result.baselineId = 'silu-ptq';
    for (const {record: r, index} of references(data, key, r => r.part === 'b')) {
      // float_acc is the *original* checkpoint on every E4 row. QAT does not
      // record an updated FP32 evaluation; activation surgery does.
      const fp32 = r.variant === 'silu-ptq' ? r.float_acc : r.float_acc_after_heal;
      const epochs = number(r.qat_epochs) ?? number(r.heal_epochs);
      append(variant(r.variant, r.variant, r.variant, {
        ...classificationFields(data, model, fp32, r.fake_acc, r.int_acc, metadata.int_eval_images),
        ...cycleFields(r.cycles?.[preset], key, preset), epochs,
        training: training(r.heal_log), originalFp32Accuracy: number(r.float_acc),
        afterSwapAccuracy: number(r.float_acc_after_swap),
        trainingConfig: own(r, 'qat_epochs') ? {epochs, learningRate: number(metadata.qat_lr),
          stepsPerEpoch: number(metadata.qat_steps_per_epoch)} : {epochs},
        sourceRecords: [pointer(key, index)]
      }));
    }
    result.notes.push('FP32는 변경한 모델의 시험셋 평가입니다. SiLU QAT의 변경 후 FP32와 학습 곡선은 기록되지 않았습니다.');
    result.metricNotes.push('회복 학습 곡선의 valAccuracy는 검증셋 지표이며, 최종 시험 정확도와 별개입니다.');
  } else if (id === 'e1-training') {
    for (const {record: r, index} of references(data, key)) {
      const images = number(r.splits?.test), logs = training(r.epochs);
      const recordedEpochs = logs.map(row => row.epoch).filter(value => value !== null);
      append(variant(r.model, LABELS[r.model] || r.model, '기준 모델 학습', {
        model: r.model, fp32Accuracy: number(r.float_acc), fp32Images: images,
        accuracyScope: scope(result.dataset, images), params: number(r.params), macs: number(r.macs),
        ...cycleFields(cost(r, preset), key, preset),
        epochs: recordedEpochs.length ? Math.max(...recordedEpochs) : null,
        training: logs, selectedEpoch: number(r.selected_epoch), finalTestAccuracy: number(r.final_test_acc),
        selection: r.selection || null, splits: copy(r.splits), sourceRecords: [pointer(key, index)]
      }));
    }
    result.baselineId = result.variants.find(r => r.id === 'resnet20_relu')?.id || result.variants[0]?.id || null;
    result.notes.push('모델별 기록된 학습 곡선입니다. 선택한 체크포인트의 FP32 시험 정확도와 마지막 epoch의 시험 정확도는 다를 수 있습니다.');
    result.metricNotes.push('trainAccuracy가 저장되지 않은 epoch는 비워 둡니다. 검증 정확도를 훈련 정확도로 대체하지 않습니다.');
  } else if (key === 'e6_pruning') {
    for (const {record: r, index} of references(data, key, r => r.model === model)) {
      const base = r.strategy === 'none';
      const variantId = base ? 'baseline' : r.strategy + '-' + r.ratio + '-align' + r.align;
      const label = base ? '프루닝 전' : r.strategy + ' · ' + r.ratio + (r.align ? ' · align ' + r.align : '');
      append(variant(variantId, label, base ? '기준 모델' : '채널 프루닝 + 회복 학습', {
        ...classificationFields(data, model, r.ft_acc, r.int8_acc, null, null),
        ...cycleFields(r[preset]?.cycles, key, preset),
        epochs: base ? 0 : number(r.ft_epochs), params: number(r.params), macs: number(r.macs),
        strategy: r.strategy, ratio: number(r.ratio), align: number(r.align), keep: copy(r.keep),
        afterPruneAccuracy: number(r.acc_after_prune), targetReached: r.target_reached ?? null,
        achievedRatio: number(r.achieved_ratio), sourceRecords: [pointer(key, index)]
      }));
    }
    result.baselineId = 'baseline';
    result.notes.push('E6의 int8_acc는 fake-quant 모델의 정확도입니다. 정수 엔진 평가와 epoch별 회복 학습 곡선은 저장되지 않았습니다.',
      '파라미터와 비용은 E6 내부의 기준·변경 모델끼리 비교합니다. E1과 파라미터 집계 범위가 다릅니다.');
    result.metricNotes.push('ratio는 uniform/aligned에서 채널 유지 비율, cost-greedy에서 목표 사이클 비율입니다. targetReached와 실제 비용을 함께 확인합니다.');
  } else if (id === 'e9-vit') {
    const {record: r, index} = references(data, key, r => r.model === 'cust_vit')[0];
    const intake = [r.intake, r.intake_strict].find(row => row?.spec === preset);
    const beforeCycles = intake?.diagnose?.cycles ?? r.alternatives?.find(row => row.preset === preset)?.cycles;
    append(variant('before', 'GELU · 변경 전', '기준 모델 + PTQ', {
      ...classificationFields(data, model, r.float_acc, r.ptq?.fake_acc, r.ptq?.int_acc, r.ptq?.int_eval_images),
      ...cycleFields(beforeCycles, key, preset), params: number(r.params), macs: number(r.macs), epochs: 0,
      fakeOnIntSubset: number(r.ptq?.fake_acc_on_int_subset), sourceRecords: [pointer(key, index)]
    }));
    if (r.after) {
      const a = r.after, epochMatch = typeof a.action === 'string' && a.action.match(/heal (\d+) epochs/);
      append(variant('after', 'ReLU · 회복 학습 후', a.action || '활성함수 교체·회복 학습', {
        ...classificationFields(data, model, a.float_acc, a.fake_acc, a.int_acc, a.int_eval_images),
        ...cycleFields(preset === 'edge-10tops' ? a.cycles : preset === 'edge-10tops-strict' ? a.cycles_strict : null, key, preset),
        epochs: epochMatch ? Number(epochMatch[1]) : null,
        fakeOnIntSubset: number(a.fake_acc_on_int_subset), sourceRecords: [pointer(key, index, 'after')]
      }));
    }
    result.baselineId = 'before';
    result.notes.push('FP32·fake-quant는 전체 시험셋, 정수 엔진은 기록된 부분집합 평가입니다. FP32와 INT8의 원시 정확도를 직접 빼지 않습니다.',
      '변경 후 비용은 edge-10tops와 edge-10tops-strict에만 기록되어 있습니다. epoch별 회복 학습 곡선은 없습니다.');
  } else if (key === 'e2_ptq_grid') {
    for (const {record: r, index} of references(data, key, r => r.model === model)) {
      append(variant(r.scheme, r.scheme, 'PTQ · ' + r.scheme, {
        ...classificationFields(data, model, r.float_acc, r.fake_acc, r.int_acc, r.int_eval_images),
        fp32OnIntSubset: number(r.float_acc_on_int_subset), fakeOnIntSubset: number(r.fake_acc_on_int_subset),
        calibration: metadata.calib || null, quantizationConfig: copy(metadata.schemes?.[r.scheme]),
        sourceRecords: [pointer(key, index)]
      }));
    }
    result.baselineId = result.variants.find(r => r.id === 'npu-default')?.id || result.variants[0]?.id || null;
    result.notes.push('양자화 방식별 정수 평가 이미지 수가 다릅니다. 같은 부분집합의 float_acc_on_int_subset과 int_acc만 양자화 손실 계산에 사용할 수 있습니다.',
      'E2의 보정 설정을 그대로 표시합니다. 다른 실험의 PTQ 수치나 비용을 가져오지 않습니다.');
  } else if (id.startsWith('e4-qat-')) {
    const {record: r, index} = references(data, key, r => r.part === 'c' && r.model === model)[0];
    append(variant('ptq', 'PTQ 기준', 'PTQ · ' + r.scheme, {
      ...classificationFields(data, model, r.float_acc, r.ptq_acc, null, null), epochs: 0,
      sourceRecords: [pointer(key, index)]
    }), variant('qat', 'QAT · ' + r.qat_epochs + ' epoch', 'QAT · ' + r.scheme, {
      ...classificationFields(data, model, null, r.qat_acc, r.int_acc, metadata.int_eval_images),
      epochs: number(r.qat_epochs), training: training(r.qat_log),
      trainingConfig: {epochs: number(r.qat_epochs), learningRate: number(metadata.qat_lr), stepsPerEpoch: number(metadata.qat_steps_per_epoch)},
      sourceRecords: [pointer(key, index)]
    }));
    result.baselineId = 'ptq';
    result.notes.push('QAT 이후에는 fake-quant와 정수 엔진 정확도가 기록되었습니다. 원본 float_acc를 변경 후 FP32 정확도로 표시하지 않습니다.');
  } else if (id === 'e4-cle-mnv2_050_relu6') {
    for (const {record: r, index} of references(data, key, r => r.part === 'a' && r.model === model)) {
      const fp32 = r.step === 'ptq' ? r.float_acc : r.float_acc_after_cle;
      append(variant(r.step, r.step, r.step + ' · ' + r.scheme, {
        ...classificationFields(data, model, fp32, r.fake_acc, r.int_acc, metadata.int_eval_images),
        epochs: number(r.qat_epochs), training: training(r.qat_log), sourceRecords: [pointer(key, index)]
      }));
    }
    result.baselineId = 'ptq';
    result.notes.push('CLE의 변경 후 FP32는 float_acc_after_cle입니다. 보정·QAT 이후 FP32가 기록되지 않은 행은 비워 둡니다.');
  } else if (id === 'e12-imagenette') {
    for (const {record: r, index} of references(data, key)) {
      const base = r.kind === 'baseline', fp32 = number(base ? r.test_acc : r.float_acc);
      const images = number(metadata.test);
      append(variant(base ? 'baseline' : r.scheme, base ? 'ResNet-20 · FP32 기준' : r.scheme,
        base ? '128×128 입력 · 기준 학습' : 'PTQ · ' + r.scheme, {
          fp32Accuracy: fp32, fakeAccuracy: number(r.fake_acc), intAccuracy: number(r.int_acc),
          fp32Images: fp32 === null ? null : images, fakeImages: number(r.fake_acc) === null ? null : images,
          intImages: number(r.int_acc) === null ? null : number(r.int_eval_images),
          accuracyScope: scope(result.dataset, images), params: number(r.params), macs: number(r.macs),
          ...cycleFields(cost(r, preset), key, preset), epochs: number(r.epochs),
          inputShape: copy(r.input_shape), selectedEpoch: number(r.selected_epoch), sourceRecords: [pointer(key, index)]
        }));
    }
    result.baselineId = 'baseline';
    result.notes.push('Imagenette-128의 독립된 학습·시험 결과입니다. CIFAR-10과는 데이터셋과 입력 해상도가 함께 달라져 정확도 변화의 원인을 분리할 수 없습니다.',
      '학습 epoch 수는 기록되었지만 epoch별 곡선은 없습니다. PTQ 방식별 비용은 기록되지 않았습니다.');
  } else if (id.startsWith('e17-')) {
    const mine = references(data, key, r => r.model === model);
    const first = mine[0], r0 = first.record;
    const convs = number(r0.conv_layers);
    const metric = (baseline, value) => ({key: 'psnr', label: 'PSNR', unit: 'dB', baseline: number(baseline), value: number(value)});
    append(variant('baseline', 'FP32 기준' + (convs === null ? '' : ' · conv ' + convs + '개'), '초해상도 기준 모델', {
      metric: metric(r0.float_psnr, r0.float_psnr), fp32Metric: number(r0.float_psnr),
      fakeMetric: null, intMetric: null, metricImages: number(r0.n_test),
      inputShape: copy(r0.input_shape), sourceRecords: [pointer(key, first.index)]
    }));
    for (const {record: r, index} of mine) {
      append(variant(r.scheme, r.scheme, 'PTQ · ' + r.scheme, {
        metric: metric(r.float_psnr, r.int_psnr), fp32Metric: number(r.float_psnr),
        fakeMetric: number(r.fake_psnr), intMetric: number(r.int_psnr), metricImages: number(r.n_test),
        calibration: metadata.calib || null, inputShape: copy(r.input_shape), sourceRecords: [pointer(key, index)]
      }));
    }
    result.baselineId = 'baseline';
    result.metricNotes.push(metadata.metric || '이미지별 PSNR (dB)', '초해상도의 출력은 픽셀입니다. 분류 정확도와 정확도 손실 %p 기준을 적용하지 않습니다.');
    result.notes.push('E17에는 비용·학습 곡선이 기록되지 않았습니다. PSNR만 해당 실험의 FP32·fake-quant·정수 경로끼리 비교합니다.');
    const depths = new Set(rows(data, key).map(r => number(r.conv_layers)).filter(v => v !== null));
    if (depths.size > 1) {
      const list = [...depths].sort((a, b) => a - b).join('개와 conv ');
      result.notes.push('같은 폭 구성을 conv ' + list + '개 두 깊이에서 측정했습니다. 깊은 쪽이 더 좋은 모델은 아니며(FP32 PSNR이 더 낮습니다), '
        + '출력 코드 불일치가 깊이에 따라 어떻게 움직이는지를 보기 위한 대조군입니다. 정수 경로의 출력 코드 불일치율은 이 카드의 PSNR과 별개 지표이며 E17 문서에 있습니다.');
    }
  }
  const base = result.variants.find(row => row.id === result.baselineId);
  if (result.task === CLASSIFICATION) {
    const samples = baselineRecord(data, model);
    for (const row of result.variants) {
      row.metric = {key: 'fp32_accuracy', label: 'FP32 Top-1', unit: '%',
        baseline: number(base?.fp32Accuracy) === null ? null : base.fp32Accuracy * 100,
        value: row.fp32Accuracy === null ? null : row.fp32Accuracy * 100};
      if (samples && (row.fp32Images !== null || row.fakeImages !== null))
        row.sourceRecords.push(pointer('e1_baselines', samples.index, 'splits/test'));
    }
    result.metricNotes.push('비교 기준은 같은 시험셋의 FP32 정확도와 기록된 동일 NPU 프리셋의 추정 사이클입니다. 비용은 하드웨어 실측 지연시간이 아닙니다.');
  }
  return result;
}

function assess(result, constraints = {}) {
  const maxDropPp = constraints.maxDropPp ?? 1, minSavingPercent = constraints.minSavingPercent ?? 10;
  if (number(maxDropPp) === null || maxDropPp < 0 || number(minSavingPercent) === null || minSavingPercent < 0)
    throw new Error('정확도 손실·비용 절감 기준에는 0 이상의 숫자가 필요합니다.');
  const variants = Array.isArray(result?.variants) ? result.variants : [];
  const base = variants.find(row => row.id === result.baselineId);
  return variants.map(row => {
    const sameAccuracy = result.task === CLASSIFICATION && base && base.accuracyScope &&
      row.accuracyScope === base.accuracyScope && number(base.fp32Images) !== null &&
      base.fp32Images > 0 && row.fp32Images === base.fp32Images &&
      number(base.fp32Accuracy) !== null && number(row.fp32Accuracy) !== null;
    const sameCycles = base && base.cycleScope && row.cycleScope === base.cycleScope &&
      number(base.cycles) !== null && base.cycles > 0 && number(row.cycles) !== null && row.cycles >= 0;
    const accuracyDropPp = sameAccuracy ? (base.fp32Accuracy - row.fp32Accuracy) * 100 : null;
    const cycleSavingPercent = sameCycles ? (1 - row.cycles / base.cycles) * 100 : null;
    let eligible = null, reason;
    if (!sameAccuracy) reason = result.task !== CLASSIFICATION ? '분류 정확도 %p 기준 적용 대상이 아닙니다.' : '동일 시험셋의 변경 후 FP32 기록이 없습니다.';
    else if (!sameCycles) reason = '동일 프리셋의 기준·변경 비용 기록이 없습니다.';
    else {
      eligible = accuracyDropPp <= maxDropPp + 1e-10 && cycleSavingPercent >= minSavingPercent - 1e-10;
      reason = eligible ? '기록된 FP32·추정 비용이 설정 기준을 충족합니다.' : '기록된 FP32·추정 비용이 설정 기준을 충족하지 않습니다.';
    }
    return {variantId: row.id, accuracyDropPp, cycleSavingPercent, eligible, reason};
  });
}

const api = Object.freeze({catalog, study, assess});
if (typeof module !== 'undefined' && module.exports) module.exports = api;
else root.NpuStudy = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
