import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import ResNet

# --- 1. Squeeze-and-Excitation (SE) 模組 ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: 全域平均池化 (將空間維度壓縮到 1x1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: 兩層 1x1 卷積組成的 MLP
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b: batch size, c: channels
        y = self.avg_pool(x)
        y = self.fc(y)
        # 將計算出的權重 (y) 乘回原始特徵圖 (x)
        return x * y

# --- 2. 結合 SE 的 Bottleneck 模組 ---
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # 實例化 SEBlock (放在最後一個 1x1 卷積之後)
        self.se = SEBlock(planes * self.expansion)

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

        # 在 Residual 加法之前，進行 Squeeze-and-Excitation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# --- 3. 建立魔改版 SE-ResNet50 的工廠函數 ---
def get_resnet50_se(num_classes):
    # 使用我們的 SEBottleneck 取代原本的 BasicBlock/Bottleneck
    model = ResNet(SEBottleneck, [3, 4, 6, 3])
    
    # 下載官方預訓練權重
    baseline_weights = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).state_dict()
    pretrained_dict = {k: v for k, v in baseline_weights.items() if not k.startswith('fc')}
    
    # 使用 strict=False 載入，讓 SEBlock 保持隨機初始化
    missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
    
    print(f"[Model Init] 成功載入官方權重。放過未初始化的 SE 參數數量: {len(missing_keys)}")
    
    # 設定最終的 FC 層
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model