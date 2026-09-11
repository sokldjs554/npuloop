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
                 widths: tuple[int, int, int] | None = None, stem_stride: int = 1):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n = (depth - 2) // 6
        w = widths or (width, 2 * width, 4 * width)
        self.stem_conv = nn.Conv2d(3, w[0], 3, stem_stride, 1, bias=False)   # stride 2 for larger inputs (Imagenette 128px)
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
        self.config = dict(arch="resnet", depth=depth, width=width, act=act, num_classes=num_classes, widths=list(w),
                           stem_stride=stem_stride)
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


class Attention(nn.Module):
    """Multi-head self-attention written so torch.fx sees every op the NPU has to run."""

    def __init__(self, dim: int, heads: int, tokens: int):
        super().__init__()
        assert dim % heads == 0
        self.h, self.d, self.n, self.c = heads, dim // heads, tokens, dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // heads) ** -0.5

    def forward(self, x):                                   # (B, N, C)
        q = self.q(x).reshape(-1, self.n, self.h, self.d).transpose(1, 2)    # (B, H, N, D)
        k = self.k(x).reshape(-1, self.n, self.h, self.d).transpose(1, 2)
        v = self.v(x).reshape(-1, self.n, self.h, self.d).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale             # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        o = torch.matmul(attn, v).transpose(1, 2).reshape(-1, self.n, self.c)
        return self.proj(o)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, tokens: int, mlp_ratio: int, act: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, tokens)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.act = make_act(act)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class ViTCIFAR(nn.Module):
    """A small ViT for 32x32 inputs: patch embedding, learned positions, mean-pooled head.

    Deliberately cls-token-free — mean pooling over tokens works as well at this scale and keeps
    the graph free of the gather/expand ops an NPU compiler would have to special-case.
    """

    def __init__(self, dim: int = 128, depth: int = 6, heads: int = 4, patch: int = 4,
                 mlp_ratio: int = 2, act: str = "gelu", num_classes: int = 10, img: int = 32):
        super().__init__()
        self.grid = img // patch
        self.tokens = self.grid ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.tokens, dim))
        self.blocks = nn.Sequential(*[TransformerBlock(dim, heads, self.tokens, mlp_ratio, act) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.config = dict(arch="vit", dim=dim, depth=depth, heads=heads, patch=patch, mlp_ratio=mlp_ratio,
                           act=act, num_classes=num_classes, img=img)
        nn.init.trunc_normal_(self.pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu"); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)     # (B, N, C)
        x = x + self.pos
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


class InceptionBlock(nn.Module):
    """Three parallel branches concatenated on the channel axis (the classic concat pattern).

    The branches see different receptive fields, so their activation ranges differ — which is
    exactly what makes concat interesting for an INT8 NPU: every input has to be requantized
    into one shared output scale.
    """

    def __init__(self, cin: int, b1: int, b3: int, b5: int, act: str):
        super().__init__()
        self.p1_conv = nn.Conv2d(cin, b1, 1, bias=False); self.p1_bn = nn.BatchNorm2d(b1); self.p1_act = make_act(act)
        self.p3_conv = nn.Conv2d(cin, b3, 3, padding=1, bias=False); self.p3_bn = nn.BatchNorm2d(b3); self.p3_act = make_act(act)
        self.p5_red = nn.Conv2d(cin, max(b5 // 2, 8), 1, bias=False); self.p5_red_bn = nn.BatchNorm2d(max(b5 // 2, 8)); self.p5_red_act = make_act(act)
        self.p5_conv = nn.Conv2d(max(b5 // 2, 8), b5, 3, padding=1, bias=False); self.p5_bn = nn.BatchNorm2d(b5); self.p5_act = make_act(act)
        self.out_channels = b1 + b3 + b5

    def forward(self, x):
        a = self.p1_act(self.p1_bn(self.p1_conv(x)))
        b = self.p3_act(self.p3_bn(self.p3_conv(x)))
        c = self.p5_act(self.p5_bn(self.p5_conv(self.p5_red_act(self.p5_red_bn(self.p5_red(x))))))
        return torch.cat([a, b, c], dim=1)


class InceptionCIFAR(nn.Module):
    """Concat-heavy CIFAR classifier: stem, three concat stages with stride-2 transitions."""

    def __init__(self, width: int = 32, act: str = "relu", num_classes: int = 10):
        super().__init__()
        w = width
        self.stem_conv = nn.Conv2d(3, w, 3, 1, 1, bias=False); self.stem_bn = nn.BatchNorm2d(w); self.stem_act = make_act(act)
        b1 = InceptionBlock(w, w // 2, w, w // 2, act)
        d1 = nn.Sequential(nn.Conv2d(b1.out_channels, 2 * w, 3, 2, 1, bias=False), nn.BatchNorm2d(2 * w), make_act(act))
        b2 = InceptionBlock(2 * w, w, 2 * w, w, act)
        d2 = nn.Sequential(nn.Conv2d(b2.out_channels, 4 * w, 3, 2, 1, bias=False), nn.BatchNorm2d(4 * w), make_act(act))
        b3 = InceptionBlock(4 * w, 2 * w, 4 * w, 2 * w, act)
        self.blocks = nn.Sequential(b1, d1, b2, d2, b3)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(b3.out_channels, num_classes)
        self.config = dict(arch="inception", width=width, act=act, num_classes=num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.blocks(x)
        return self.fc(self.flatten(self.pool(x)))


def build_model(config: dict) -> nn.Module:
    cfg = dict(config)
    arch = cfg.pop("arch")
    if arch == "resnet":
        cfg["widths"] = tuple(cfg["widths"]) if cfg.get("widths") else None
        return ResNetCIFAR(**cfg)
    if arch == "mobilenetv2":
        return MobileNetV2CIFAR(**cfg)
    if arch == "vit":
        return ViTCIFAR(**cfg)
    if arch == "inception":
        return InceptionCIFAR(**cfg)
    raise ValueError(arch)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
