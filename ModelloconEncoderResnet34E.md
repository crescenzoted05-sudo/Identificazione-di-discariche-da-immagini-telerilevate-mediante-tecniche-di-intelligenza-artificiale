
import sys
print(sys.executable)

import torch
print(torch.__version__)
print(torch.cuda.is_available())
import os
import random
import time
import gc
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import segmentation_models_pytorch as smp

# =========================
# PATH DATASET GIÀ SPLITTATO
# =========================
train_x_dir = r"C:\Users\Crescenzo\Desktop\Split_512Augmentation\train\images"
train_y_dir = r"C:\Users\Crescenzo\Desktop\Split_512Augmentation\train\masks"

val_x_dir = r"C:\Users\Crescenzo\Desktop\Split_512Augmentation\val\images"
val_y_dir = r"C:\Users\Crescenzo\Desktop\Split_512Augmentation\val\masks"

test_x_dir = r"C:\Users\Crescenzo\Desktop\Split_512Augmentation\test\images"
test_y_dir = r"C:\Users\Crescenzo\Desktop\Split_512Augmentation\test\masks"

train_files = sorted(os.listdir(train_x_dir))
val_files = sorted(os.listdir(val_x_dir))
test_files = sorted(os.listdir(test_x_dir))

print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))


# =========================
# DATASET
# =========================
class BuildingsDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_names, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_names = file_names
        self.augment = augment

    def __getitem__(self, index):
        file_name = self.file_names[index]

        image_path = os.path.join(self.images_dir, file_name)
        mask_path = os.path.join(self.masks_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, [512, 512], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [512, 512], interpolation=InterpolationMode.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0).float()

        return image, mask

    def __len__(self):
        return len(self.file_names)


train_dataset = BuildingsDataset(train_x_dir, train_y_dir, train_files, augment=True)
valid_dataset = BuildingsDataset(val_x_dir, val_y_dir, val_files, augment=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)


import segmentation_models_pytorch as smp

# =========================
# MODELLO U-NET CON ENCODER RESNET34
# =========================
def build_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    return model


# =========================
# LOSS E METRICHE
# =========================
class Jaccard(nn.Module):
    def __init__(self):
        super(Jaccard, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou


class IoUScore(nn.Module):
    def __init__(self):
        super(IoUScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.47).float()
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou


epochs = 20
learning_rate = 0.00001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = build_model().to(device)

pos_weight = torch.tensor([5.0], device=device)
loss1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss2 = Jaccard()
iou_metrics = IoUScore()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# =========================
# TRAIN
# =========================
import os

if os.path.exists("checkpoint_last_512_Resnet34.pt"):
    checkpoint = torch.load("checkpoint_last_512_Resnet34.pt", map_location=device)
    history = checkpoint.get("history", {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": []
    })

    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    old_history = checkpoint.get("history", {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": []
    })
    min_loss = checkpoint.get("best_val_loss", np.inf)
    best_miou = checkpoint.get("best_val_miou", 0.0)

else:
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    old_history = {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": []
    }
    min_loss = np.inf
    best_miou = 0.0


def train(epochs, model, train_loader, val_loader, loss1, loss2, iou_metrics, optimizer):
    global min_loss, best_miou, old_history, start_epoch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_losses = []
    val_losses = []
    train_iou = []
    val_iou = []

    fit_time = time.time()

    for e in range(epochs):
        since = time.time()
        running_loss = 0.0
        train_iou_score = 0.0

        current_epoch = start_epoch + e + 1

        # =========================
        # TRAIN LOOP
        # =========================
        model.train()
        for image, mask in tqdm(train_loader, desc=f"Train Epoch {current_epoch}/{start_epoch + epochs}"):
            image = image.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            mask_pred = model(image)

            loss = loss1(mask_pred, mask) + loss2(mask_pred, mask)
            iou_score = iou_metrics(mask_pred, mask)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_iou_score += iou_score.item()

        # =========================
        # VALIDATION LOOP
        # =========================
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_iou = 0.0

        with torch.no_grad():
            for image, mask in tqdm(val_loader, desc=f"Val Epoch {current_epoch}/{start_epoch + epochs}"):
                image = image.to(device)
                mask = mask.to(device)

                mask_pred = model(image)

                loss = loss1(mask_pred, mask) + loss2(mask_pred, mask)
                iou_score = iou_metrics(mask_pred, mask)

                epoch_val_loss += loss.item()
                epoch_val_iou += iou_score.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_train_iou = train_iou_score / len(train_loader)
        avg_val_iou = epoch_val_iou / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_iou.append(avg_train_iou)
        val_iou.append(avg_val_iou)

        # =========================
        # SALVATAGGIO BEST VAL LOSS
        # =========================
        if min_loss > avg_val_loss:
            print(f"Loss Decreasing.. {min_loss:.3f} >> {avg_val_loss:.3f}")
            min_loss = avg_val_loss

            torch.save({
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": {
                    "train_loss": old_history["train_loss"] + train_losses,
                    "val_loss": old_history["val_loss"] + val_losses,
                    "train_miou": old_history["train_miou"] + train_iou,
                    "val_miou": old_history["val_miou"] + val_iou,
                },
                "best_val_loss": min_loss,
                "best_val_miou": best_miou,
            }, "checkpoint_best_512_Resnet34.pt")

        # =========================
        # SALVATAGGIO BEST VAL MIOU
        # =========================
        print(f"best_miou attuale={best_miou:.3f} | val_mIoU corrente={avg_val_iou:.3f}")
        if avg_val_iou > best_miou:
            print(f"mIoU Increasing.. {best_miou:.3f} >> {avg_val_iou:.3f}")
            best_miou = avg_val_iou

            torch.save({
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": {
                    "train_loss": old_history["train_loss"] + train_losses,
                    "val_loss": old_history["val_loss"] + val_losses,
                    "train_miou": old_history["train_miou"] + train_iou,
                    "val_miou": old_history["val_miou"] + val_iou,
                },
                "best_val_loss": min_loss,
                "best_val_miou": best_miou,
            }, "checkpoint_best_miou_512_Resnet34.pt")

        print(
            f"Epoch: {current_epoch} | "
            f"Train Loss: {avg_train_loss:.3f} | "
            f"Val Loss: {avg_val_loss:.3f} | "
            f"Train mIoU: {avg_train_iou:.3f} | "
            f"Val mIoU: {avg_val_iou:.3f} | "
            f"Time: {(time.time()-since)/60:.2f}m"
        )

    history = {
        "train_loss": old_history["train_loss"] + train_losses,
        "val_loss": old_history["val_loss"] + val_losses,
        "train_miou": old_history["train_miou"] + train_iou,
        "val_miou": old_history["val_miou"] + val_iou,
    }

    print(f"Total time: {(time.time()-fit_time)/60:.2f} m")

    torch.save({
        "epoch": start_epoch + epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_loss": min_loss,
        "best_val_miou": best_miou,
    }, "checkpoint_last_512_Resnet34.pt")

    return history


print("start_epoch =", start_epoch)
print("best_val_loss =", min_loss)
print("best_val_miou =", best_miou)
print("history salvata =", len(old_history["train_loss"]))

# =========================
# RUN LOOP
# =========================

#history = train(10, model, train_loader, val_loader, loss1, loss2, iou_metrics, optimizer)
#print("Training finito")
#print("Best mIoU finale:", best_miou)

#print("len train_loss =", len(history["train_loss"]))
#print("len val_loss =", len(history["val_loss"]))
#print("len train_miou =", len(history["train_miou"]))
#print("len val_miou =", len(history["val_miou"]))

# =========================
# RUN SINGOLO 20 epochs
# =========================
#if __name__ == "__main__":
    #history = train(epochs, model, train_loader, val_loader, loss1, loss2, iou_metrics, optimizer)
    #print("Training finito")

# =========================
# GRAFICI TRAINING DA CHECKPOINT
# =========================
import matplotlib.pyplot as plt

USE_BEST_FOR_PLOTS = False

plot_checkpoint_path = (
    "checkpoint_best_miou_512_Resnet34.pt"
    if USE_BEST_FOR_PLOTS
    else "checkpoint_last_512_Resnet34.pt"
)

if os.path.exists(plot_checkpoint_path):
    checkpoint_plot = torch.load(plot_checkpoint_path, map_location=device)
    history_plot = checkpoint_plot["history"]

    print("Grafici caricati da:", plot_checkpoint_path)
    print("Epoca salvata nel checkpoint:", checkpoint_plot["epoch"])
    print("Best val mIoU nel checkpoint:", checkpoint_plot.get("best_val_miou"))
    print("Best val loss nel checkpoint:", checkpoint_plot.get("best_val_loss"))

    # Grafico Loss
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(history_plot["train_loss"]) + 1),
        history_plot["train_loss"],
        marker="o",
        label="Train Loss"
    )
    plt.plot(
        range(1, len(history_plot["val_loss"]) + 1),
        history_plot["val_loss"],
        marker="o",
        label="Validation Loss"
    )
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Andamento della Loss durante il training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_loss_checkpoint_512_Resnet34.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Grafico mIoU
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(history_plot["train_miou"]) + 1),
        history_plot["train_miou"],
        marker="o",
        label="Train mIoU"
    )
    plt.plot(
        range(1, len(history_plot["val_miou"]) + 1),
        history_plot["val_miou"],
        marker="o",
        label="Validation mIoU"
    )
    plt.xlabel("Epoca")
    plt.ylabel("mIoU")
    plt.title("Andamento della mIoU durante il training")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_miou_checkpoint_512_Resnet34.png", dpi=300, bbox_inches="tight")
    plt.show()

else:
    print("Checkpoint per i grafici non trovato:", plot_checkpoint_path)

# =========================
# DATASET DI TEST
# =========================
class TestBuildingsDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_names):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_names = file_names

    def __getitem__(self, index):
        file_name = self.file_names[index]

        image_path = os.path.join(self.images_dir, file_name)
        mask_path = os.path.join(self.masks_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, [512, 512], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [512, 512], interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0).float()

        return image, mask, file_name

    def __len__(self):
        return len(self.file_names)

test_dataset = TestBuildingsDataset(test_x_dir, test_y_dir, test_files)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

# =========================
# TEST COMPLETO SU TUTTO IL TEST SET
# =========================
if os.path.exists("checkpoint_best_miou_512_Resnet34.pt"):
    checkpoint_test = torch.load("checkpoint_best_miou_512_Resnet34.pt", map_location=device)
    model.load_state_dict(checkpoint_test["model_state_dict"])
    print("Caricato checkpoint_best_miou_512_Resnet34.pt per il test finale")
else:
    print("checkpoint_best_miou_512_Resnet34.pt non trovato, uso il modello corrente")
model.eval()
test_loss = 0.0
test_iou = 0.0

with torch.no_grad():
    for image, mask, file_name in tqdm(test_loader, desc="Test completo"):
        image = image.to(device)
        mask = mask.to(device)

        pred = model(image)

        loss = loss1(pred, mask) + loss2(pred, mask)
        iou_score = iou_metrics(pred, mask)

        test_loss += loss.item()
        test_iou += iou_score.item()

avg_test_loss = test_loss / len(test_loader)
avg_test_iou = test_iou / len(test_loader)

print(f"TEST FINALE | Loss: {avg_test_loss:.3f} | mIoU: {avg_test_iou:.3f}")


# =========================
# 2 ESEMPI POSITIVI + 1 NEGATIVO DAL TEST SET
# =========================
model.eval()

positive_target = 2
negative_target = 1

positive_shown = 0
negative_shown = 0

with torch.no_grad():
    for image, mask, file_name in test_loader:
        image = image.to(device).float()
        mask = mask.to(device).float()

        pred = model(image)
        pred = torch.sigmoid(pred)

        current_batch_size = image.shape[0]

        for b in range(current_batch_size):
            print("===================================")
            print("File:", file_name[b])
            print("min pred =", pred[b].min().item())
            print("max pred =", pred[b].max().item())
            print("mean pred =", pred[b].mean().item())
            print("pixel > 0.5 =", (pred[b] > 0.5).sum().item())
            print("pixel > 0.4 =", (pred[b] > 0.4).sum().item())
            print("pixel > 0.3 =", (pred[b] > 0.3).sum().item())
            print("pixel > 0.2 =", (pred[b] > 0.2).sum().item())
            print("pixel > 0.1 =", (pred[b] > 0.1).sum().item())

            mask_np_check = mask[b].cpu().squeeze(0).numpy()
            is_positive = np.sum(mask_np_check > 0) > 0

            if is_positive and positive_shown >= positive_target:
                continue
            if (not is_positive) and negative_shown >= negative_target:
                continue

            pred_bin = (pred[b] > 0.51).float()

            img_np = image[b].cpu().permute(1, 2, 0).numpy()
            mask_np = mask_np_check
            pred_np = pred_bin.cpu().squeeze(0).numpy()

            plt.figure(figsize=(18, 5))

            plt.subplot(1, 4, 1)
            plt.imshow(img_np)
            plt.title(f"Immagine\n{file_name[b]}")
            plt.axis("off")

            plt.subplot(1, 4, 2)
            plt.imshow(mask_np, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 4, 3)
            plt.imshow(pred_np, cmap="gray")
            plt.title("Predizione binaria")
            plt.axis("off")

            plt.subplot(1, 4, 4)
            plt.imshow(img_np)
            plt.imshow(pred_np, cmap="Reds", alpha=0.5)
            plt.title("Overlay predizione")
            plt.axis("off")

            plt.tight_layout()

            if is_positive:
                positive_shown += 1
                save_name = f"test_positive_example512_Resnet34{positive_shown}.png"
            else:
                negative_shown += 1
                save_name = f"test_negative_example512_Resnet34{negative_shown}.png"

            plt.savefig(save_name, dpi=300, bbox_inches="tight")
            print("Salvato:", save_name)

            plt.show()
            plt.close()

            if positive_shown >= positive_target and negative_shown >= negative_target:
                break

        if positive_shown >= positive_target and negative_shown >= negative_target:
            break

