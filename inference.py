import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import TestDataset, get_dataloaders
from models import (
    get_resnet50_baseline,
    get_resnet50_cbam,
    get_resnet50_se,
    get_resnext50_cbam,
)

# --- 參數設定 ---
OUTPUT_DIR = "./outputs/[your_exp_name]"  # 請替換成你的實驗資料夾名稱
DATA_DIR = "./data"
TEST_DIR = "./data/test"
BATCH_SIZE = 128

# 自動推導輸出資料夾路徑
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SAVE_PATH = os.path.join(OUTPUT_DIR, "prediction.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    sns.heatmap(cm, annot=False, cmap="Blues", cbar=True, square=True)
    plt.title("Confusion Matrix on Validation Set", fontsize=16, pad=20)
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
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
    val_loader = dataloaders["val"]

    # 2. 準備模型並載入權重
    print(f"\n>>> 正在載入模型權重: {CHECKPOINT_PATH}")
    # model = get_resnet50_baseline(num_classes=num_classes)
    # model = get_resnet50_cbam(num_classes=num_classes)
    # model = get_resnet50_se(num_classes=len(class_names))
    model = get_resnext50_cbam(num_classes=num_classes)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model = model.to(device)
    model.eval()

    # 3. 測試集前處理 (與 Val 一致)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_dataset = TestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8
    )

    # 4. 開始推論 Test Set
    print("\n>>> 開始推論 Test Set 並產生 submission CSV...")
    results = []
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="Test Inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for name, pred in zip(img_names, preds):
                results.append(
                    {"image_name": os.path.splitext(name)[0], "pred_label": pred.item()}
                )

    # 5. 存成 CSV
    df = pd.DataFrame(results)
    df.to_csv(SAVE_PATH, index=False)
    print(f"    - 完成！CSV 已儲存至 {SAVE_PATH}，共 {len(results)} 筆資料。")

    # ==========================================
    # 6. 自動產出報告圖表
    # ==========================================
    cm_save_path = os.path.join(OUTPUT_DIR, "report_confusion_matrix.png")

    plot_confusion_matrix(model, val_loader, cm_save_path)

    print("\n>>> 所有推論與圖表生成任務已完美結束！請查看 output 資料夾。")


if __name__ == "__main__":
    main()
