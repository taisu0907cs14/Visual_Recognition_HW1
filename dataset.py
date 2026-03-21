import os
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset

# --- 1. 自定義的 ImageFolder (強制數字排序) ---
class NumericImageFolder(datasets.ImageFolder):
    def find_classes(self, directory: str):
        """
        覆寫原本的 find_classes，強制使用「數字大小」來排序資料夾。
        這樣資料夾 '2' 就會乖乖排在 '10' 前面，避免標籤對應錯誤。
        """
        # 抓取所有子資料夾名稱
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]
        
        # 關鍵：強制轉成整數來排序
        classes.sort(key=lambda x: int(x))
        
        # 建立對應字典
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

# --- 2. 訓練與驗證資料集載入函數 ---
def get_dataloaders(data_dir, batch_size=32, num_workers=8):
    # ImageNet 標準常規化參數
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 手動建構的綜合資料增強清單 (取代 TrivialAugmentWide)
    # RandomChoice 會從這個清單中「隨機挑選 1 個」來對當前圖片執行
    custom_augment_space = transforms.RandomChoice([
        transforms.Lambda(lambda x: x),  # 1. 什麼都不做 (保持原圖)
        transforms.ColorJitter(brightness=(0.1, 1.9)),  # 2. 亮度
        transforms.ColorJitter(contrast=(0.1, 1.9)),    # 3. 對比度
        transforms.ColorJitter(saturation=(0.1, 1.9)),  # 4. 飽和度
        transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=1.0),  # 5. 銳利度
        transforms.RandomAffine(degrees=0, shear=(-20, 20)),  # 6. 錯切 (扭曲變形)
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # 7. 平移
        transforms.RandomRotation(degrees=(-30, 30)),  # 8. 旋轉
        transforms.RandomAutocontrast(p=1.0),  # 9. 自動對比
        transforms.RandomEqualize(p=1.0),      # 10. 直方圖均衡化
        # 如果你的任務對色彩細節非常敏感 (例如微小瑕疵)，建議把下面兩個註解掉：
        # transforms.RandomPosterize(bits=4, p=1.0),   # 11. 色調分離 (減少色彩位元)
        # transforms.RandomSolarize(threshold=128, p=1.0)  # 12. 曝光過度 (反轉亮部像素)
    ])

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            
            # 插入我們自定義的增強清單
            custom_augment_space,
            
            transforms.ToTensor(),
            normalize,
            
            # 隨機抹除 (這是一種非常強的正則化手段，必須放在 ToTensor 之後)
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    }

    # 使用我們自定義的 NumericImageFolder
    image_datasets = {
        x: NumericImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in ['train', 'val']
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                       shuffle=(x == 'train'), num_workers=num_workers)
        for x in ['train', 'val']
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    # 每次啟動時印出前 10 個類別，讓你安心確認排序是正常的
    print(f"[Dataset] 成功載入！確認類別排序 (前10個): {class_names[:10]}")
    
    return dataloaders, dataset_sizes, class_names

# --- 3. 專為推論 (Inference) 設計的測試資料集載入器 ---
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 加上 sorted() 確保讀取順序固定，這在 Debug 時非常重要
        self.image_names = sorted([
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        # 確保圖片轉換為 RGB 格式 (避免灰階圖片導致維度錯誤)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name