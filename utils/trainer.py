import os
import torch
from torch.utils.tensorboard import SummaryWriter

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                device, num_epochs=50, output_dir='./outputs'):
    
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    log_path = os.path.join(output_dir, 'log.txt')
    
    # --- 新增：計算並記錄模型參數量 ---
    # p.numel() 會回傳該層 tensor 的元素總數
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_info = (
        f"{'='*30}\n"
        f"模型參數量統計:\n"
        f"- 總參數量 (Total): {total_params:,}\n"
        f"- 可訓練參數 (Trainable): {trainable_params:,}\n"
        f"{'='*30}"
    )
    
    # 印在終端機
    print(param_info)
    
    # 寫入 log.txt
    with open(log_path, 'a') as f:
        f.write(param_info + '\n\n')
    # ----------------------------------
    
    best_acc = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        header = f'Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.8f}'
        print(f'\n{header}\n' + '-'*10)
        with open(log_path, 'a') as f:
            f.write(f'\n{header}\n' + '-'*10 + '\n')

        writer.add_scalar('Learning_Rate', current_lr, epoch)

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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            res = f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
            print(res)
            with open(log_path, 'a') as f:
                f.write(res + '\n')

            writer.add_scalar(f'Loss/{phase}_epoch', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}_epoch', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                # print(f'>>> Best Model Saved (Acc: {best_acc:.4f})')
                res = f'>>> Best Model Saved (Acc: {best_acc:.4f})'
                print(res)
                with open(log_path, 'a') as f:
                    f.write(res + '\n')

        # 每個 Epoch 結束後動態儲存 Last Model
        torch.save(model.state_dict(), os.path.join(output_dir, 'last_model.pth'))
        scheduler.step()

    writer.close()
    print(f'\n訓練完成！結果儲存在: {output_dir}')
    return model