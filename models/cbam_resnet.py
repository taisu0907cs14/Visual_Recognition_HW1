import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNeXt50_32X4D_Weights
from torchvision.models.resnet import ResNet


# --- 1. 通道注意力模組 (Channel Attention) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的 MLP 層
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# --- 2. 空間注意力模組 (Spatial Attention) ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿著 Channel 維度作池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# --- 3. 結合 CBAM 的 Bottleneck 模組 ---
class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(CBAMBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            groups=groups,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, stride=1, bias=False
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # 實例化 CBAM
        self.ca = ChannelAttention(planes * self.expansion)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 在 Residual 加法之前，依序進行 Channel 與 Spatial Attention
        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# --- 4. 建立魔改版 ResNet50 的工廠函數 ---
def get_resnet50_cbam(num_classes):
    # 使用我們的 CBAMBottleneck 取代原本的 BasicBlock/Bottleneck
    model = ResNet(CBAMBottleneck, [3, 4, 6, 3])

    # 下載官方預訓練權重
    baseline_weights = models.resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V1
    ).state_dict()

    # 剔除與任務不符的 FC 層參數
    pretrained_dict = {
        k: v for k, v in baseline_weights.items() if not k.startswith("fc")
    }

    # 關鍵：使用 strict=False 載入。
    # 這會載入所有卷積層的預訓練特徵，但放過我們新增的 CBAM 模組，讓它們保持隨機初始化
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)

    print(
        f"[Model Init] 成功載入官方權重。放過未初始化的 CBAM 參數數量: {len(missing_keys)}"
    )

    # 設定最終的 FC 層
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnext50_cbam(num_classes):
    """
    建立帶有 CBAM 注意力機制的 ResNeXt50_32x4d 模型
    """
    # ResNeXt 的精髓：將原本的一大坨卷積，拆分成 32 個獨立的群組 (groups=32)
    # 每個群組的通道寬度為 4 (width_per_group=4)
    model = ResNet(CBAMBottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)

    # 載入 PyTorch 官方的 ResNeXt50 預訓練權重
    baseline_weights = models.resnext50_32x4d(
        weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1
    ).state_dict()

    # 剔除最後的分類層 FC
    pretrained_dict = {
        k: v for k, v in baseline_weights.items() if not k.startswith("fc")
    }

    # 使用 strict=False 載入 (放過我們新增的 CBAM 層)
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)

    print(
        f"[Model Init] 成功載入 ResNeXt50 官方權重。放過未初始化的 CBAM 參數數量: {len(missing_keys)}"
    )

    # 設定你自己的 100 類 FC 層
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
