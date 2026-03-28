import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# 匯入自定義模組
from dataset import get_dataloaders
from models import get_resnet50_baseline, get_resnet50_cbam, get_resnet50_se
from utils import train_model

# --- 設定超參數 ---
EXP_NAME = 'baseline_res50'
DATA_DIR = './data'
BATCH_SIZE = 128
NUM_EPOCHS = 100
WARMUP_EPOCHS = 10
LEARNING_RATE = 1e-4

# 動態設定輸出資料夾
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join('./outputs', f"{current_time}_{EXP_NAME}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. 資料
    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    
    # 2. 模型
    model = get_resnet50_baseline(num_classes=len(class_names))
    # model = get_resnet50_cbam(num_classes=len(class_names))
    # model = get_resnet50_se(num_classes=len(class_names))
    model = model.to(device)

    # 3. 優化器與排程
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

    s1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
    s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(NUM_EPOCHS - WARMUP_EPOCHS))
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[s1, s2], milestones=[WARMUP_EPOCHS])

    # 4. 執行
    train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
        device=device, num_epochs=NUM_EPOCHS, output_dir=OUTPUT_DIR
    )

if __name__ == '__main__':
    main()