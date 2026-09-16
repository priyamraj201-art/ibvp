"""
OSNet: Omni-Scale Network for Person Re-Identification.
Reference:
    Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV 2019.
    Zhou et al. Learning Generalisable Omni-Scale Representations
    for Person Re-Identification. TPAMI 2021.
"""

from __future__ import division, absolute_import
import os
import sys
import errno
import warnings
import urllib.request
import ssl
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from loguru import logger

# Fix Windows OpenMP collision
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__all__ = ["OSNet", "osnet_x0_25"]

# Direct download URLs for OSNet-x0.25 pretrained weights
WEIGHTS_URLS = [
    # MSMT17 Re-ID fine-tuned weights (paulosantiago HF mirror)
    "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt",
    # KaiyangZhou HF repo mirror
    "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_imagenet.pth",
]


class ConvLayer(nn.Module):
    """Convolution layer (conv + bn + relu)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        IN=False,
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution: 1x1 (linear) + depthwise 3x3 (nonlinear)."""

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation="sigmoid",
        reduction=16,
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        mid_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(in_channels, mid_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(mid_channels, num_gates, 1, bias=True)
        if gate_activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == "linear":
            self.gate_activation = None
        else:
            raise RuntimeError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlock(nn.Module):
    """Omni-scale feature learning block with 4 scale streams and unified aggregation gate."""

    def __init__(
        self,
        in_channels,
        out_channels,
        IN=False,
        bottleneck_reduction=4,
        **kwargs,
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        residual = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = x3 + residual
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out, inplace=True)


class OSNet(nn.Module):
    """
    Omni-Scale Network (OSNet) architecture.
    """

    def __init__(
        self,
        num_classes=1000,
        blocks=None,
        layers=None,
        channels=None,
        feature_dim=512,
        IN=False,
        **kwargs,
    ):
        super(OSNet, self).__init__()
        if blocks is None:
            blocks = [OSBlock, OSBlock, OSBlock]
        if layers is None:
            layers = [2, 2, 2]
        if channels is None:
            channels = [16, 64, 96, 128]

        num_blocks = len(blocks)
        assert num_blocks == len(layers) == len(channels) - 1

        # Stem layer
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Main stages
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN,
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
            IN=IN,
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
            IN=IN,
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Feature dimensionality projection to 512-D
        self.fc = self._construct_fc_layer(
            feature_dim, channels[3], dropout_p=None
        )

        # Optional classification head
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(feature_dim, num_classes)
        else:
            self.classifier = None

        self._init_params()

    def _make_layer(
        self,
        block,
        layer,
        in_channels,
        out_channels,
        reduce_spatial_size,
        IN=False,
    ):
        layers = []
        layers.append(block(in_channels, out_channels, IN=IN))
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))
        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2),
                )
            )
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or int(fc_dims) < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_featuremaps=False):
        """
        Forward pass.
        Returns 512-D L2-normalized feature representation.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        return v


def download_pretrained_weights(save_path: str) -> bool:
    """Download pretrained OSNet-x0.25 weights with SSL handling."""
    ctx = ssl._create_unverified_context()
    for url in WEIGHTS_URLS:
        try:
            logger.info(f"Downloading OSNet-x0.25 weights from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                data = resp.read()
                if len(data) > 100_000:  # Must be at least ~100KB
                    with open(save_path, "wb") as f:
                        f.write(data)
                    logger.info(f"OSNet-x0.25 weights saved ({len(data)} bytes) to {save_path}")
                    return True
        except Exception as e:
            logger.warning(f"Failed download from {url}: {e}")
    return False


def load_pretrained_weights(model: nn.Module, weight_path: str):
    """Safely load OSNet-x0.25 state dict with key mapping."""
    if not os.path.isfile(weight_path):
        return False

    try:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model_dict = model.state_dict()
        new_state_dict = OrderedDict()
        matched = 0

        for k, v in state_dict.items():
            clean_k = k
            if clean_k.startswith("module."):
                clean_k = clean_k[7:]
            if clean_k.startswith("model."):
                clean_k = clean_k[6:]

            if clean_k in model_dict and model_dict[clean_k].size() == v.size():
                new_state_dict[clean_k] = v
                matched += 1

        if matched > 0:
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Loaded {matched} layers into OSNet-x0.25 from {weight_path}")
            return True
        else:
            logger.warning(f"No matching layers found in {weight_path}")
            return False
    except Exception as e:
        logger.warning(f"Failed to load OSNet weights from {weight_path}: {e}")
        return False


def osnet_x0_25(
    num_classes: int = 1000,
    pretrained: bool = True,
    weights_dir: str = "pretrained",
    **kwargs,
) -> OSNet:
    """
    Build lightweight OSNet-x0.25 model (2.2 MB, 512-dim embedding).
    """
    model = OSNet(
        num_classes=num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        feature_dim=512,
        **kwargs,
    )

    if pretrained:
        # Resolve target weights path
        bytetrack_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        weights_dir_abs = os.path.join(bytetrack_root, weights_dir)
        os.makedirs(weights_dir_abs, exist_ok=True)
        cached_weights = os.path.join(weights_dir_abs, "osnet_x0_25.pth")

        if not os.path.isfile(cached_weights):
            download_pretrained_weights(cached_weights)

        if os.path.isfile(cached_weights):
            load_pretrained_weights(model, cached_weights)
        else:
            logger.warning("OSNet-x0.25 initialized with random weights (fallback mode).")

    return model
