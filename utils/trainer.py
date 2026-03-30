import os

import torch
from torch.utils.tensorboard import SummaryWriter


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=50,
    output_dir="./outputs",
    start_epoch=0,
    best_acc=0.0,
    global_step=0,
):

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "logs"))
    log_path = os.path.join(output_dir, "log.txt")

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
    with open(log_path, "a") as f:
        f.write(param_info + "\n\n")
    # ----------------------------------

    for epoch in range(start_epoch, start_epoch + num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        # 顯示絕對 Epoch 數字，不要印出 1/90 這種容易混淆的格式
        header = f"Epoch {epoch+1} | LR: {current_lr:.8f}"
        print(f"\n{header}\n" + "-" * 10)
        with open(log_path, "a") as f:
            f.write(f"\n{header}\n" + "-" * 10 + "\n")

        writer.add_scalar("Learning_Rate", current_lr, epoch)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        global_step += 1  # 🌟 這裡會無縫接續上一階段的步數
                        if batch_idx % 20 == 0:
                            writer.add_scalar(
                                "Loss/train_iter", loss.item(), global_step
                            )

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            res = f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"
            print(res)
            with open(log_path, "a") as f:
                f.write(res + "\n")

            # 🌟 x 軸使用絕對的 epoch 數值
            writer.add_scalar(f"Loss/{phase}_epoch", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/{phase}_epoch", epoch_acc, epoch)

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc  # 🌟 完美繼承，不會被重置
                torch.save(
                    model.state_dict(), os.path.join(output_dir, "best_model.pth")
                )
                res = f">>> Best Model Saved (Acc: {best_acc:.4f})"
                print(res)
                with open(log_path, "a") as f:
                    f.write(res + "\n")

        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pth"))
        scheduler.step()

    writer.close()

    # 🌟 訓練結束後，回傳當前的狀態給外部
    return model, best_acc, global_step
