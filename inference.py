import torch
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset, get_dataloaders
from models import get_resnet50_baseline, get_resnet50_cbam, get_resnet50_se

# --- 參數設定 ---
# 請填入你訓練產出的最佳權重路徑
OUTPUT_DIR = './outputs/20260329_161307_cbam_freeze_thaw_label_smoothing'
DATA_DIR = './data'
TEST_DIR = './data/test'
BATCH_SIZE = 128

# 自動推導輸出資料夾路徑
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, 'best_model.pth')
SAVE_PATH = os.path.join(OUTPUT_DIR, 'prediction.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 繪圖附屬函數 (自動產出 Report 圖表)
# ==========================================
def plot_training_curve(log_dir, save_path):
    print("\n>>> 正在讀取 TensorBoard Logs 繪製訓練曲線...")
    if not os.path.exists(log_dir):
        print(f"    - [警告] 找不到 log 資料夾: {log_dir}，跳過曲線繪製。")
        return

    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if 'events' in f]
    if not event_files:
        print("    - [警告] log 資料夾內沒有 events 檔案，跳過曲線繪製。")
        return

    ea = EventAccumulator(event_files[0], size_guidance={'scalars': 0})
    ea.Reload()

    try:
        epochs = [e.step for e in ea.Scalars('Loss/train_epoch')]
        train_loss = [e.value for e in ea.Scalars('Loss/train_epoch')]
        val_loss = [e.value for e in ea.Scalars('Loss/val_epoch')]
        val_acc = [e.value for e in ea.Scalars('Accuracy/val_epoch')]
        train_acc = [e.value for e in ea.Scalars('Accuracy/train_epoch')]
    except Exception as e:
        print(f"    - [警告] 讀取 log 數值失敗 ({e})，跳過曲線繪製。")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)

    # 左軸：Loss
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', color='tab:red', fontsize=12)
    ax1.plot(epochs, train_loss, color='tab:red', linestyle='-', alpha=0.6, label='Train Loss')
    ax1.plot(epochs, val_loss, color='tab:red', linestyle='-', linewidth=2, label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    # 右軸：Accuracy
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
    ax2.plot(epochs, train_acc, color='tab:blue', linestyle='--', alpha=0.6, label='Train Acc')
    ax2.plot(epochs, val_acc, color='tab:blue', linestyle='-', linewidth=2, label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 合併圖例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')

    plt.title('Training Curve: Loss & Accuracy', fontsize=14, pad=15)
    fig.tight_layout()
    plt.savefig(save_path)
    print(f"    - 訓練曲線已儲存至: {save_path}")
    plt.close()

def plot_confusion_matrix(model, val_loader, save_path):
    print("\n>>> 正在 Validation Set 上進行推論並繪製混淆矩陣...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="CM Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10), dpi=150)
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, square=True)
    plt.title('Confusion Matrix on Validation Set', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"    - 混淆矩陣已儲存至: {save_path}")
    plt.close()

# ==========================================
# 主程式
# ==========================================
def main():
    # 1. 取得正確的類別數量與 Validation Dataloader
    dataloaders, _, class_names = get_dataloaders(DATA_DIR, BATCH_SIZE)
    num_classes = len(class_names)
    val_loader = dataloaders['val']

    # 2. 準備模型並載入權重
    print(f"\n>>> 正在載入模型權重: {CHECKPOINT_PATH}")
    # model = get_resnet50_baseline(num_classes=num_classes)
    model = get_resnet50_cbam(num_classes=num_classes)
    # model = get_resnet50_se(num_classes=len(class_names))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model = model.to(device)
    model.eval()

    # 3. 測試集前處理 (與 Val 一致)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = TestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # 4. 開始推論 Test Set
    print("\n>>> 開始推論 Test Set 並產生 submission CSV...")
    results = []
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="Test Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for name, pred in zip(img_names, preds):
                results.append({
                    'image_name': os.path.splitext(name)[0], 
                    'pred_label': pred.item()
                })

    # 5. 存成 CSV
    df = pd.DataFrame(results)
    df.to_csv(SAVE_PATH, index=False)
    print(f"    - 完成！CSV 已儲存至 {SAVE_PATH}，共 {len(results)} 筆資料。")

    # ==========================================
    # 6. 自動產出報告圖表
    # ==========================================
    log_dir = os.path.join(OUTPUT_DIR, 'logs')
    curve_save_path = os.path.join(OUTPUT_DIR, 'report_training_curve.png')
    cm_save_path = os.path.join(OUTPUT_DIR, 'report_confusion_matrix.png')

    plot_training_curve(log_dir, curve_save_path)
    plot_confusion_matrix(model, val_loader, cm_save_path)
    
    print("\n>>> 所有推論與圖表生成任務已完美結束！請查看 output 資料夾。")

if __name__ == '__main__':
    main()