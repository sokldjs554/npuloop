"""Small CIFAR-10 CNNs written to be torch.fx-traceable and quantization-tool friendly.

Design rules (they matter for the analyzer / quantizer later):
  * every activation is an explicit nn.Module (no functional calls) so passes can swap them
  * no in-place ops
  * residual adds are plain `a + b` so they show up as operator.add nodes in fx
  * conv -> bn -> act ordering everywhere (so BN folding is mechanical)
"""
from __future__ import annotations
import torch
import torch.nn as nn

ACTS = {
    "relu": lambda: nn.ReLU(),
    "relu6": lambda: nn.ReLU6(),
    "silu": lambda: nn.SiLU(),
    "gelu": lambda: nn.GELU(),
    "hswish": lambda: nn.Hardswish(),
    "lrelu": lambda: nn.LeakyReLU(0.1),
}


def make_act(name: str) -> nn.Module:
    return ACTS[name]()


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, act: str):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.act1 = make_act(act)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act2 = make_act(act)
        self.has_shortcut = stride != 1 or cin != cout
        if self.has_shortcut:
            self.sc_conv = nn.Conv2d(cin, cout, 1, stride, 0, bias=False)
            self.sc_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        identity = self.sc_bn(self.sc_conv(x)) if self.has_shortcut else x
        return self.act2(y + identity)


class ResNetCIFAR(nn.Module):
    """ResNet-(6n+2) for CIFAR (He et al. 2016). depth=20 -> n=3, widths 16/32/64."""

    def __init__(self, depth: int = 20, width: int = 16, act: str = "relu", num_classes: int = 10,
                 widths: tuple[int, int, int] | None = None):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n = (depth - 2) // 6
        w = widths or (width, 2 * width, 4 * width)
        self.stem_conv = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(w[0])
        self.stem_act = make_act(act)
        blocks = []
        cin = w[0]
        for stage, cout in enumerate(w):
            for k in range(n):
                stride = 2 if (stage > 0 and k == 0) else 1
                blocks.append(BasicBlock(cin, cout, stride, act))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(cin, num_classes)
        self.config = dict(arch="resnet", depth=depth, width=width, act=act, num_classes=num_classes, widths=list(w))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.blocks(x)
        return self.fc(self.flatten(self.pool(x)))


class InvertedResidual(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int, expand: int, act: str):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.expand = expand != 1
        if self.expand:
            self.pw_conv = nn.Conv2d(cin, hidden, 1, 1, 0, bias=False)
            self.pw_bn = nn.BatchNorm2d(hidden)
            self.pw_act = make_act(act)
        self.dw_conv = nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden)
        self.dw_act = make_act(act)
        self.proj_conv = nn.Conv2d(hidden, cout, 1, 1, 0, bias=False)
        self.proj_bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        y = x
        if self.expand:
            y = self.pw_act(self.pw_bn(self.pw_conv(y)))
        y = self.dw_act(self.dw_bn(self.dw_conv(y)))
        y = self.proj_bn(self.proj_conv(y))
        return y + x if self.use_res else y


class MobileNetV2CIFAR(nn.Module):
    """MobileNetV2 adapted to 32x32 inputs (strides of the first stages set to 1)."""

    def __init__(self, width_mult: float = 1.0, act: str = "relu6", num_classes: int = 10,
                 setting=None, last_channels: int = 1280):
        super().__init__()
        setting = setting or [  # t, c, n, s  (CIFAR variant)
            [1, 16, 1, 1], [6, 24, 2, 1], [6, 32, 3, 2], [6, 64, 4, 2],
            [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
        def c8(v):
            return max(8, int(round(v * width_mult / 8)) * 8)
        cin = c8(32)
        self.stem_conv = nn.Conv2d(3, cin, 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(cin)
        self.stem_act = make_act(act)
        blocks = []
        for t, c, n, s in setting:
            cout = c8(c)
            for i in range(n):
                blocks.append(InvertedResidual(cin, cout, s if i == 0 else 1, t, act))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        last = c8(last_channels) if width_mult > 1.0 else last_channels
        self.head_conv = nn.Conv2d(cin, last, 1, 1, 0, bias=False)
        self.head_bn = nn.BatchNorm2d(last)
        self.head_act = make_act(act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(last, num_classes)
        self.config = dict(arch="mobilenetv2", width_mult=width_mult, act=act, num_classes=num_classes,
                           setting=setting, last_channels=last_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.blocks(x)
        x = self.head_act(self.head_bn(self.head_conv(x)))
        return self.fc(self.flatten(self.pool(x)))


def build_model(config: dict) -> nn.Module:
    cfg = dict(config)
    arch = cfg.pop("arch")
    if arch == "resnet":
        cfg["widths"] = tuple(cfg["widths"]) if cfg.get("widths") else None
        return ResNetCIFAR(**cfg)
    if arch == "mobilenetv2":
        return MobileNetV2CIFAR(**cfg)
    raise ValueError(arch)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
