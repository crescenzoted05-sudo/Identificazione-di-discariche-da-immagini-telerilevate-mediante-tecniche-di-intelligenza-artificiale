
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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0)


# =========================
# MODELLO U-NET ULTRA LEGGERO
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode='bilinear', dropout=0.0):
        super(UpBlock, self).__init__()

        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(
                in_channels - out_channels,
                in_channels - out_channels,
                kernel_size=2,
                stride=2
            )
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported up_sample_mode")

        self.double_conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, out_classes=1, up_sample_mode='bilinear'):
        super(UNet, self).__init__()

        self.down_conv1 = DownBlock(3, 16, dropout=0.0)
        self.down_conv2 = DownBlock(16, 32, dropout=0.0)
        self.down_conv3 = DownBlock(32, 64, dropout=0.1)
        self.down_conv4 = DownBlock(64, 128, dropout=0.1)

        self.double_conv = DoubleConv(128, 256, dropout=0.2)

        self.up_conv4 = UpBlock(128 + 256, 128, up_sample_mode, dropout=0.1)
        self.up_conv3 = UpBlock(64 + 128, 64, up_sample_mode, dropout=0.1)
        self.up_conv2 = UpBlock(32 + 64, 32, up_sample_mode, dropout=0.0)
        self.up_conv1 = UpBlock(16 + 32, 16, up_sample_mode, dropout=0.0)

        self.conv_last = nn.Conv2d(16, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)

        x = self.double_conv(x)

        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)

        x = self.conv_last(x)
        x = torch.sigmoid(x)
        return x


# =========================
# LOSS E METRICHE
# =========================
class Jaccard(nn.Module):
    def __init__(self):
        super(Jaccard, self).__init__()

    def forward(self, inputs, targets, smooth=1):
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
        inputs = (inputs > 0.5).float()
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

model = UNet().to(device)

loss1 = nn.BCELoss()
loss2 = Jaccard()
iou_metrics = IoUScore()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# =========================
# TRAIN
# =========================
import os

if os.path.exists("checkpoint_last_512.pt"):
    checkpoint = torch.load("checkpoint_last_512.pt", map_location=device)
    history = checkpoint.get("history", {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": []
    })

    model = UNet().to(device)
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
    model = UNet().to(device)
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
            }, "checkpoint_best_512.pt")

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
            }, "checkpoint_best_miou_512.pt")

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
    }, "checkpoint_last_512.pt")

    return history


print("start_epoch =", start_epoch)
print("best_val_loss =", min_loss)
print("best_val_miou =", best_miou)
print("history salvata =", len(old_history["train_loss"]))

# =========================
# RUN LOOP
# =========================

history = train(20, model, train_loader, val_loader, loss1, loss2, iou_metrics, optimizer)
print("Training finito")
print("Best mIoU finale:", best_miou)

# =========================
# RUN SINGOLO 20 epochs
# =========================
#if __name__ == "__main__":
    #history = train(epochs, model, train_loader, val_loader, loss1, loss2, iou_metrics, optimizer)
    #print("Training finito")

# =========================
# GRAFICI TRAINING
# =========================
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], marker="o", label="Train Loss")
plt.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.title("Andamento della Loss durante il training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_loss_512.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(history["train_miou"]) + 1), history["train_miou"], marker="o", label="Train mIoU")
plt.plot(range(1, len(history["val_miou"]) + 1), history["val_miou"], marker="o", label="Validation mIoU")
plt.xlabel("Epoca")
plt.ylabel("mIoU")
plt.title("Andamento della mIoU durante il training")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_miou_512.png", dpi=300, bbox_inches="tight")
plt.show()

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
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# =========================
# TEST COMPLETO SU TUTTO IL TEST SET
# =========================
if os.path.exists("checkpoint_best_miou_512.pt"):
    checkpoint_test = torch.load("checkpoint_best_miou_512.pt", map_location=device)
    model.load_state_dict(checkpoint_test["model_state_dict"])
    print("Caricato checkpoint_best_miou_512.pt per il test finale")
else:
    print("checkpoint_best_miou.pt non trovato, uso il modello corrente")
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
# IMMAGINE ESTERNA
# =========================
input_image_path = r"C:\Users\Crescenzo\Desktop\immagine di prova\immaginergb\662_0_488.png"
input_mask_path = r"C:\Users\Crescenzo\Desktop\immagine di prova\Immaginemask\662_0_488.png"

print(input_image_path)
print(input_mask_path)
print(os.path.isfile(input_image_path))
print(os.path.isfile(input_mask_path))

image_pil = Image.open(input_image_path).convert("RGB")
mask_pil = Image.open(input_mask_path).convert("L")

image_resized = TF.resize(image_pil, [512, 512], interpolation=InterpolationMode.BILINEAR)
mask_resized = TF.resize(mask_pil, [512, 512], interpolation=InterpolationMode.NEAREST)

image_tensor = TF.to_tensor(image_resized).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    pred = model(image_tensor)
    pred_bin = (pred > 0.5).float()

img_np = np.array(image_resized)
mask_np = np.array(mask_resized)
pred_np = pred_bin.squeeze(0).squeeze(0).cpu().numpy()

plt.figure(figsize=(18, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_np)
plt.title("Immagine input")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask_np, cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(pred_np, cmap="gray")
plt.title("Predizione")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(img_np)
plt.imshow(pred_np, cmap="Reds", alpha=0.5)
plt.title("Overlay predizione")
plt.axis("off")

plt.tight_layout()
plt.show()

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
        pred_bin = (pred > 0.5).float()

        current_batch_size = image.shape[0]

        for b in range(current_batch_size):
            mask_np_check = mask[b].cpu().squeeze(0).numpy()
            is_positive = np.sum(mask_np_check > 0) > 0

            if is_positive and positive_shown >= positive_target:
                continue
            if (not is_positive) and negative_shown >= negative_target:
                continue

            img_np = image[b].cpu().permute(1, 2, 0).numpy()
            mask_np = mask_np_check
            pred_np = pred_bin[b].cpu().squeeze(0).numpy()

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
                save_name = f"test_positive_example512_{positive_shown}.png"
            else:
                negative_shown += 1
                save_name = f"test_negative_example512_{negative_shown}.png"

            plt.savefig(save_name, dpi=300, bbox_inches="tight")
            print("Salvato:", save_name)

            plt.show()
            plt.close()

            if positive_shown >= positive_target and negative_shown >= negative_target:
                break

        if positive_shown >= positive_target and negative_shown >= negative_target:
            break

print("Positivi salvati:", positive_shown)
print("Negativi salvati:", negative_shown)

