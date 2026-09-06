import json, os, subprocess, sys
import torch


def _tiny_ckpt(tmp_path):
    from npuloop.zoo import ResNetCIFAR
    torch.manual_seed(0)
    m = ResNetCIFAR(depth=8, width=8).eval()
    p = tmp_path / "tiny.pt"
    torch.save({"config": m.config, "state_dict": m.state_dict()}, p)
    return p


def test_cli_cost_and_lint(tmp_path):
    ckpt = _tiny_ckpt(tmp_path)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, "-m", "npuloop", "cost", str(ckpt), "--spec", "tiny-1tops", "--json", str(tmp_path / "c.json")],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "TOTAL cycles" in out.stdout
    d = json.load(open(tmp_path / "c.json"))
    assert d["total_cycles"] > 0 and d["spec"]["name"] == "tiny-1tops"
    out = subprocess.run([sys.executable, "-m", "npuloop", "lint", str(ckpt), "--spec", "edge-10tops", "--json", str(tmp_path / "l.json")],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "NPU readiness report" in out.stdout
    assert 0 <= json.load(open(tmp_path / "l.json"))["scores"]["overall"] <= 100
