"""Restrict checkpoint deserialization; never auto-retry with unrestricted pickle."""
import ast
from pathlib import Path
import pickle
import pytest
import torch
from npuloop.zoo import build_model, load_checkpoint
from npuloop.cli import _load

ROOT=Path(__file__).resolve().parents[1]

def _write_probe(path):
    Path(path).write_text('object callback executed')
    return {}

class _ProbeObject:
    def __init__(self, path): self.path=path
    def __reduce__(self): return _write_probe, (self.path,)

@pytest.mark.parametrize('loader',[load_checkpoint, _load])
def test_checkpoint_does_not_execute_custom_object_callback(loader,tmp_path):
    marker=tmp_path/'marker'; path=tmp_path/'object.pt'
    torch.save(_ProbeObject(str(marker)),path)
    try:
        loader(path)
    except Exception:
        pass
    assert not marker.exists(), 'Unrestricted pickle object callback executed'

@pytest.mark.parametrize('loader',[load_checkpoint, _load])
def test_checkpoint_rejects_malformed_schema(loader,tmp_path):
    path=tmp_path/'bad.pt';torch.save({'config':{},'state_dict':{}},path)
    with pytest.raises(ValueError,match='checkpoint'):
        loader(path)

@pytest.mark.parametrize('name',['resnet20_relu.pt','cust_inception.pt'])
def test_bundled_weights_load_on_cpu_without_changing_values(name):
    path=ROOT/'examples/quickstart'/name
    raw=torch.load(path,weights_only=True,map_location='cpu')
    m=load_checkpoint(path)
    for k,v in m.state_dict().items():
        assert v.device.type=='cpu'
        assert torch.equal(v,raw['state_dict'][k])

@pytest.mark.parametrize('filename',['npuloop/cli.py','npuloop/zoo/train.py','npuloop/zoo/train_sr.py'])
def test_all_checkpoint_load_calls_use_restricted_cpu_loading(filename):
    tree=ast.parse((ROOT/filename).read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='load' and isinstance(node.func.value,ast.Name) and node.func.value.id=='torch':
            kw={k.arg:ast.literal_eval(k.value) for k in node.keywords if k.arg in ('weights_only','map_location')}
            assert kw=={'weights_only':True,'map_location':'cpu'}, (filename,node.lineno,kw)

@pytest.mark.parametrize('task',['classification','super_resolution'])
def test_current_training_state_resumes_with_restricted_loader(task,tmp_path):
    import numpy as np
    from npuloop.zoo.data import CIFAR10NPZ, SRPairs
    from npuloop.zoo.train import fit
    from npuloop.zoo.train_sr import fit_sr
    rng=np.random.default_rng(0);data=tmp_path/'data.npz'
    np.savez(data,x_train=rng.integers(0,256,(80,16,16,3),dtype=np.uint8),
             y_train=np.arange(80)%10,x_test=rng.integers(0,256,(20,16,16,3),dtype=np.uint8),
             y_test=np.arange(20)%10,classes=np.array([str(i) for i in range(10)]),val_per_class=1,pad=2)
    out=tmp_path/'run'
    if task=='classification':
        ds=CIFAR10NPZ(str(data));config={'arch':'resnet','depth':8,'width':16,'act':'relu'};trainer=fit;metric='test_acc'
    else:
        ds=SRPairs(str(data));config={'arch':'espcn','lr_size':8,'scale':2,'feat':8};trainer=fit_sr;metric='test_psnr'
    original=build_model(config)
    first=trainer(original,ds,epochs=2,bs=16,steps_per_epoch=4,out=str(out))
    state=torch.load(out/'state.pt',weights_only=True,map_location='cpu')
    assert 'rng' in state and 'torch_rng' in state and state['epoch']==2
    restored=build_model(config)
    second=trainer(restored,ds,epochs=2,bs=16,steps_per_epoch=4,out=str(out),resume=True)
    assert second[metric]==pytest.approx(first[metric],abs=1e-6)
    for key,value in original.state_dict().items():
        assert torch.equal(value,restored.state_dict()[key])
