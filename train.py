import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

# 匯入自定義模組
from dataset import get_dataloaders
from models import get_resnet50_baseline, get_resnet50_cbam, get_resnet50_se, get_resnext50_cbam
from utils import train_model

# --- 設定超參數 ---
EXP_NAME = 'cbam_freeze_thaw_label_smoothing_weighted'  # 這個名字會用來命名輸出資料夾，建議包含模型架構與訓練策略的關鍵字，方便日後回顧
DATA_DIR = './data'
BATCH_SIZE = 128

# 將原本的 100 Epochs 拆分為兩個階段
PHASE1_EPOCHS = 10  # 階段一：只訓練頭部與注意力模組
PHASE2_EPOCHS = 90  # 階段二：微調全網路
PHASE1_LR = 1e-3    # 階段一學習率可以大一點
PHASE2_LR = 1e-4    # 階段二學習率必須小，以保護預訓練權重

# 動態設定輸出資料夾
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = os.path.join('./outputs', f"{current_time}_{EXP_NAME}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. 資料
    dataloaders, dataset_sizes, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    
    # 2. 模型 (這裡以 CBAM 為例，你也可以換成 SE)
    # model = get_resnet50_baseline(num_classes=len(class_names))
    # model = get_resnet50_se(num_classes=len(class_names))
    # model = get_resnext50_cbam(num_classes=len(class_names))
    model = get_resnet50_cbam(num_classes=len(class_names))
    model = model.to(device)

    # 3. 恢復為最乾淨的 CrossEntropyLoss (移除 Label Smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ==========================================
    # 階段一：凍結主幹 (Freeze Backbone)
    # ==========================================
    print("\n" + "="*40)
    print(f">>> 啟動階段一：凍結主幹，只訓練注意力與分類層 ({PHASE1_EPOCHS} Epochs)")
    print("="*40)

    # 凍結所有參數
    for param in model.parameters():
        param.requires_grad = False

    # 只解凍我們想訓練的層：FC 層與注意力模組 (包含 'se', 'ca', 'sa' 的命名)
    for name, param in model.named_parameters():
        if 'fc' in name or 'se' in name or 'ca' in name or 'sa' in name:
            param.requires_grad = True

    # 階段一的 Optimizer：只傳入 requires_grad=True 的參數
    optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR, weight_decay=0.05)
    scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=PHASE1_EPOCHS)

    # 🌟 執行階段一，並把回傳的狀態存起來
    model, current_best_acc, current_step = train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer1, scheduler1,
        device=device, num_epochs=PHASE1_EPOCHS, output_dir=OUTPUT_DIR
    )

    # ==========================================
    # 階段二：全面解凍 (Thaw / Unfreeze)
    # ==========================================
    print("\n" + "="*40)
    print(f">>> 啟動階段二：全面解凍，以較小學習率微調全網路 ({PHASE2_EPOCHS} Epochs)")
    print("="*40)

    # 解凍所有參數
    for param in model.parameters():
        param.requires_grad = True

    # 階段二的 Optimizer：傳入所有參數
    optimizer2 = optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=0.05)

    # 階段二依然保留 Warmup 機制，避免剛解凍時的梯度震盪破壞權重
    s1 = optim.lr_scheduler.LinearLR(optimizer2, start_factor=0.01, total_iters=5)
    s2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=(PHASE2_EPOCHS - 5))
    scheduler2 = optim.lr_scheduler.SequentialLR(optimizer2, schedulers=[s1, s2], milestones=[5])

    # 🌟 執行階段二，並把階段一的狀態餵進去
    train_model(
        model, dataloaders, dataset_sizes, criterion, optimizer2, scheduler2,
        device=device, num_epochs=PHASE2_EPOCHS, output_dir=OUTPUT_DIR,
        start_epoch=PHASE1_EPOCHS,       # 從第 10 個 Epoch 繼續畫 TensorBoard
        best_acc=current_best_acc,       # 繼承階段一的最高準確率
        global_step=current_step         # 繼承階段一的訓練步數
    )

if __name__ == '__main__':
    main()