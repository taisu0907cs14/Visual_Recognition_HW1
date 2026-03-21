import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter  # 新增：引入 TensorBoard 支援

# 1. 設定裝置 (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 新增：設定輸出資料夾 ---
output_dir = './outputs/2603201101'
os.makedirs(output_dir, exist_ok=True)

# 2. 資料前處理 (Data Augmentation & Normalization)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]),
}

# --- 準備資料集 ---
data_dir = './data'

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True, num_workers=4),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=128, shuffle=False, num_workers=4)
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f"偵測到的類別: {class_names}")
print(f"類別總數: {num_classes}")

# --- 建立模型 ---
# 3. 建立標準 ResNet50 模型並載入預訓練權重
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# 4. 修改最後的全連接層 (FC Layer) 以符合你的任務類別數
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# 5. 設定 Loss Function, Optimizer (AdamW) 與串接式 Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05) 

num_epochs = 50
warmup_epochs = 5

# --- 定義兩個階段的 Scheduler ---
# 階段 1: Linear Warmup (從 0.01 倍的初始 LR 開始，到 1.0 倍結束)
scheduler_warmup = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
)

# 階段 2: Cosine Annealing (在 Warmup 結束後開始下降)
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=(num_epochs - warmup_epochs)
)

# 使用 SequentialLR 將兩者串接，milestones 指定在哪個 Epoch 切換
scheduler = optim.lr_scheduler.SequentialLR(
    optimizer, 
    schedulers=[scheduler_warmup, scheduler_cosine], 
    milestones=[warmup_epochs]
)

# 6. 更新後的訓練迴圈
def train_model(model, criterion, optimizer, scheduler, num_epochs=10, output_dir='./outputs'):
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    best_acc = 0.0 
    global_step = 0 

    for epoch in range(num_epochs):
        # 1. 在 Epoch 開始時取得並顯示目前的 LR
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.8f}')
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                        global_step += 1
                        if batch_idx % 20 == 0:
                            writer.add_scalar('Loss/train_iter', loss.item(), global_step)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            writer.add_scalar(f'Loss/{phase}_epoch', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}_epoch', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

        # --- 每個 Epoch 結束後更新一次 Scheduler ---
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(output_dir, 'last_model.pth'))
    writer.close()
    return model

# 開始訓練 (記得把 scheduler 傳進去)
model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs, output_dir=output_dir)