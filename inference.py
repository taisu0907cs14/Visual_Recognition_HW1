import torch
import pandas as pd
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TestDataset, get_dataloaders
from models import get_resnet50_baseline, get_resnet50_cbam, get_resnet50_se

# --- 參數設定 ---
# 請填入你訓練產出的資料夾路徑
CHECKPOINT_PATH = './outputs/20260328_174657_cbam_freeze_thaw/best_model.pth'
DATA_DIR = './data'
TEST_DIR = './data/test'
SAVE_PATH = './outputs/20260328_174657_cbam_freeze_thaw/prediction.csv'
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. 取得正確的類別數量 (從 get_dataloaders 拿，確保與訓練一致)
    _, _, class_names = get_dataloaders(DATA_DIR)
    num_classes = len(class_names)

    # 2. 準備模型並載入權重
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

    # 4. 開始推論
    results = []
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="Inference"):
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
    print(f"完成！CSV 已儲存至 {SAVE_PATH}，共 {len(results)} 筆資料。")

if __name__ == '__main__':
    main()