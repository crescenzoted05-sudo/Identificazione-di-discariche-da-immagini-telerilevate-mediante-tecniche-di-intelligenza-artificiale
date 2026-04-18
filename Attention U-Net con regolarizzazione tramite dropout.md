import sys
print(sys.executable)

import os
import random
import time
import gc
import numpy as np
from PIL import Image
from multiprocessing import freeze_support

import torch
print(torch.__version__)
print(torch.cuda.is_available())

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, ColorJitter, RandomCrop
from tqdm import tqdm
import matplotlib.pyplot as plt


# =========================
# SEED PER RIPRODUCIBILITA'
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# DATASET CON AUGMENTAZIONI AVANZATE
# =========================
class BuildingsDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_names, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_names = file_names
        self.augment = augment
        self.color_jitter = ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        )

    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_path = os.path.join(self.images_dir, file_name)
        mask_path = os.path.join(self.masks_dir, file_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, [512, 512], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [512, 512], interpolation=InterpolationMode.NEAREST)

        if self.augment:
            # Flip orizzontale
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Flip verticale
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Rotazione 90/180/270
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

            # Rotazione fine
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

            # Color jitter
            if random.random() > 0.5:
                image = self.color_jitter(image)

            # Random crop + resize
            if random.random() > 0.5:
                i, j, h, w = RandomCrop.get_params(image, output_size=(384, 384))
                image = TF.resized_crop(
                    image, i, j, h, w, [512, 512],
                    interpolation=InterpolationMode.BILINEAR
                )
                mask = TF.resized_crop(
                    mask, i, j, h, w, [512, 512],
                    interpolation=InterpolationMode.NEAREST
                )

            # Gaussian blur leggero
            if random.random() > 0.7:
                image = TF.gaussian_blur(image, kernel_size=3)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0).float()

        image = TF.normalize(
            image,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return image, mask

    def __len__(self):
        return len(self.file_names)


# =========================
# ATTENTION GATE
# =========================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# =========================
# BUILDING BLOCKS
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, dropout=dropout)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, up_sample_mode='bilinear', dropout=0.0):
        super().__init__()

        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.attention = AttentionGate(
            F_g=in_channels,
            F_l=skip_channels,
            F_int=max(skip_channels // 2, 1)
        )

        self.double_conv = DoubleConv(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)

        # Allineamento dimensioni in caso di mismatch
        if x.shape[-2:] != skip_input.shape[-2:]:
            x = F.interpolate(x, size=skip_input.shape[-2:], mode='bilinear', align_corners=True)

        skip = self.attention(g=x, x=skip_input)
        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


# =========================
# U-NET CON ATTENTION GATES
# =========================
class UNet(nn.Module):
    def __init__(self, out_classes=1, up_sample_mode='bilinear'):
        super().__init__()

        self.down_conv1 = DownBlock(3, 16, dropout=0.0)
        self.down_conv2 = DownBlock(16, 32, dropout=0.0)
        self.down_conv3 = DownBlock(32, 64, dropout=0.1)
        self.down_conv4 = DownBlock(64, 128, dropout=0.1)

        self.bottleneck = DoubleConv(128, 256, dropout=0.2)

        self.up_conv4 = UpBlock(256, 128, skip_channels=128, up_sample_mode=up_sample_mode, dropout=0.1)
        self.up_conv3 = UpBlock(128, 64, skip_channels=64, up_sample_mode=up_sample_mode, dropout=0.1)
        self.up_conv2 = UpBlock(64, 32, skip_channels=32, up_sample_mode=up_sample_mode, dropout=0.0)
        self.up_conv1 = UpBlock(32, 16, skip_channels=16, up_sample_mode=up_sample_mode, dropout=0.0)

        self.conv_last = nn.Conv2d(16, out_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x, skip1 = self.down_conv1(x)
        x, skip2 = self.down_conv2(x)
        x, skip3 = self.down_conv3(x)
        x, skip4 = self.down_conv4(x)

        x = self.bottleneck(x)

        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        x = self.up_conv1(x, skip1)

        return self.conv_last(x)


# =========================
# LOSS FUNCTIONS
# =========================
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inter = (inputs * targets).sum()
        dice = (2.0 * inter + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_w=0.4, dice_w=0.4, focal_w=0.2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=0.8, gamma=2.0)
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.focal_w = focal_w

    def forward(self, inputs, targets):
        return (
            self.bce_w * self.bce(inputs, targets)
            + self.dice_w * self.dice(inputs, targets)
            + self.focal_w * self.focal(inputs, targets)
        )


# =========================
# METRICHE
# =========================
class Metrics(nn.Module):
    def __init__(self, threshold=0.47):
        super().__init__()
        self.threshold = threshold

    def forward(self, inputs, targets, smooth=1):
        inputs = (torch.sigmoid(inputs) > self.threshold).float()
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        tp = (inputs * targets).sum()
        fp = inputs.sum() - tp
        fn = targets.sum() - tp

        iou = (tp + smooth) / (tp + fp + fn + smooth)
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)

        return iou, dice, precision, recall


# =========================
# EARLY STOPPING
# =========================
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  [EarlyStopping] Nessun miglioramento ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0


# =========================
# GRAFICI
# =========================
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(history["val_loss"], marker="o", label="Val Loss")
    axes[0].set_title("Loss durante il training")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_miou"], marker="o", label="Train mIoU")
    axes[1].plot(history["val_miou"], marker="o", label="Val mIoU")
    axes[1].set_title("mIoU durante il training")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("mIoU")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("grafico_training_v2.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================
# TEST SET
# =========================
class TestBuildingsDataset(Dataset):
    def __init__(self, images_dir, masks_dir, file_names):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.file_names = file_names

    def __getitem__(self, index):
        fn = self.file_names[index]
        image = Image.open(os.path.join(self.images_dir, fn)).convert("RGB")
        mask = Image.open(os.path.join(self.masks_dir, fn)).convert("L")

        image = TF.resize(image, [512, 512], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [512, 512], interpolation=InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = (mask > 0).float()

        image = TF.normalize(
            image,
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )

        return image, mask, fn

    def __len__(self):
        return len(self.file_names)


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


# =========================
# TRAINING LOOP
# =========================
def train_loop(
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    metrics_fn,
    optimizer,
    scheduler,
    scaler,
    early_stop,
    device,
    checkpoint_last,
    checkpoint_best_loss,
    checkpoint_best_miou,
    start_epoch,
    old_history,
    min_loss,
    best_miou
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_losses, val_losses = [], []
    train_mious, val_mious = [], []

    fit_time = time.time()

    for e in range(epochs):
        since = time.time()
        current_epoch = start_epoch + e + 1

        # ---------- TRAIN ----------
        model.train()
        run_loss = 0.0
        run_iou = 0.0

        for image, mask in tqdm(train_loader, desc=f"[Train] Epoch {current_epoch}/{start_epoch + epochs}"):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(image)
                loss = criterion(pred, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            iou, _, _, _ = metrics_fn(pred.detach(), mask)
            run_loss += loss.item()
            run_iou += iou.item()

        scheduler.step(current_epoch)

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        val_prec = 0.0
        val_rec = 0.0

        with torch.no_grad():
            for image, mask in tqdm(val_loader, desc=f"[Val]   Epoch {current_epoch}/{start_epoch + epochs}"):
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pred = model(image)
                    loss = criterion(pred, mask)

                iou, dice, prec, rec = metrics_fn(pred, mask)

                val_loss += loss.item()
                val_iou += iou.item()
                val_dice += dice.item()
                val_prec += prec.item()
                val_rec += rec.item()

        avg_train_loss = run_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_iou = run_iou / len(train_loader)
        avg_val_iou = val_iou / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_prec = val_prec / len(val_loader)
        avg_val_rec = val_rec / len(val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_mious.append(avg_train_iou)
        val_mious.append(avg_val_iou)

        elapsed = (time.time() - since) / 60

        print(
            f"Epoch {current_epoch:03d} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Train mIoU: {avg_train_iou:.4f} | "
            f"Val mIoU: {avg_val_iou:.4f} | "
            f"Val Dice: {avg_val_dice:.4f} | "
            f"Precision: {avg_val_prec:.4f} | "
            f"Recall: {avg_val_rec:.4f} | "
            f"Time: {elapsed:.2f}m"
        )

        full_history = {
            "train_loss": old_history["train_loss"] + train_losses,
            "val_loss": old_history["val_loss"] + val_losses,
            "train_miou": old_history["train_miou"] + train_mious,
            "val_miou": old_history["val_miou"] + val_mious,
        }

        # aggiorno prima i best veri
        if avg_val_loss < min_loss:
            print(f"  [Best Loss] {min_loss:.4f} -> {avg_val_loss:.4f}")
            min_loss = avg_val_loss

        if avg_val_iou > best_miou:
            print(f"  [Best mIoU] {best_miou:.4f} -> {avg_val_iou:.4f}")
            best_miou = avg_val_iou

        base_ckpt = {
            "epoch": current_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": full_history,
            "best_val_loss": min_loss,
            "best_val_miou": best_miou,
        }

        # salva best loss
        if avg_val_loss == min_loss:
            torch.save(base_ckpt, checkpoint_best_loss)

        # salva best mIoU
        if avg_val_iou == best_miou:
            torch.save(base_ckpt, checkpoint_best_miou)

        # salva ultimo ogni 10 epoche
        if current_epoch % 10 == 0:
            torch.save(base_ckpt, checkpoint_last)

        early_stop.step(avg_val_iou)
        if early_stop.should_stop:
            print(f"  [EarlyStopping] Stop a epoca {current_epoch}")
            torch.save(base_ckpt, checkpoint_last)
            break

    torch.save(base_ckpt, checkpoint_last)

    print(f"\nTotal time: {(time.time() - fit_time) / 60:.2f} m | Best mIoU: {best_miou:.4f}")
    return full_history, min_loss, best_miou


def main():
    set_seed(42)

    # =========================
    # PATH DATASET
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
    train_dataset = BuildingsDataset(train_x_dir, train_y_dir, train_files, augment=True)
    valid_dataset = BuildingsDataset(val_x_dir, val_y_dir, val_files, augment=False)

    # num_workers=0 per evitare crash multiprocessing su Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # =========================
    # SETUP
    # =========================
    epochs = 20
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = UNet().to(device)
    criterion = CombinedLoss(bce_w=0.4, dice_w=0.4, focal_w=0.2)
    metrics = Metrics(threshold=0.47)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    early_stop = EarlyStopping(patience=8, min_delta=1e-4)

    # =========================
    # CHECKPOINT
    # =========================
    CHECKPOINT_LAST = "checkpoint_last_512_v2.pt"
    CHECKPOINT_BEST_LOSS = "checkpoint_best_loss_512_v2.pt"
    CHECKPOINT_BEST_MIOU = "checkpoint_best_miou_512_v2.pt"

    if os.path.exists(CHECKPOINT_LAST):
        ckpt = torch.load(CHECKPOINT_LAST, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt.get("scheduler_state_dict", scheduler.state_dict()))
        start_epoch = ckpt.get("epoch", 0)
        old_history = ckpt.get(
            "history",
            {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}
        )
        min_loss = ckpt.get("best_val_loss", np.inf)
        best_miou = ckpt.get("best_val_miou", 0.0)
        print(f"Ripreso da epoca {start_epoch}, best mIoU={best_miou:.4f}")
    else:
        start_epoch = 0
        old_history = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}
        min_loss = np.inf
        best_miou = 0.0

    print(f"start_epoch={start_epoch} | best_loss={min_loss:.4f} | best_miou={best_miou:.4f}")

    # =========================
    # TRAIN
    # =========================
    history, min_loss, best_miou = train_loop(
        epochs=epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        metrics_fn=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stop=early_stop,
        device=device,
        checkpoint_last=CHECKPOINT_LAST,
        checkpoint_best_loss=CHECKPOINT_BEST_LOSS,
        checkpoint_best_miou=CHECKPOINT_BEST_MIOU,
        start_epoch=start_epoch,
        old_history=old_history,
        min_loss=min_loss,
        best_miou=best_miou
    )

    print("Training completato!")

    # =========================
    # GRAFICI
    # =========================
    plot_history(history)

    # =========================
    # TEST
    # =========================
    test_dataset = TestBuildingsDataset(test_x_dir, test_y_dir, test_files)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if os.path.exists(CHECKPOINT_BEST_MIOU):
        ckpt = torch.load(CHECKPOINT_BEST_MIOU, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Caricato {CHECKPOINT_BEST_MIOU} | best_miou={ckpt['best_val_miou']:.4f}")

    model.eval()
    test_loss = 0.0
    test_iou = 0.0
    test_dice = 0.0
    test_prec = 0.0
    test_rec = 0.0

    with torch.no_grad():
        for image, mask, _ in tqdm(test_loader, desc="Test"):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(image)
                loss = criterion(pred, mask)

            iou, dice, prec, rec = metrics(pred, mask)
            test_loss += loss.item()
            test_iou += iou.item()
            test_dice += dice.item()
            test_prec += prec.item()
            test_rec += rec.item()

    n = len(test_loader)
    print(
        f"\nTEST FINALE | "
        f"Loss: {test_loss / n:.4f} | "
        f"mIoU: {test_iou / n:.4f} | "
        f"Dice: {test_dice / n:.4f} | "
        f"Precision: {test_prec / n:.4f} | "
        f"Recall: {test_rec / n:.4f}"
    )

    # =========================
    # VISUALIZZAZIONE ESEMPI
    # =========================
    model.eval()
    pos_shown = 0
    neg_shown = 0
    pos_target = 2
    neg_target = 2

    with torch.no_grad():
        for image, mask, file_names in test_loader:
            image = image.to(device)
            mask = mask.to(device)

            pred = torch.sigmoid(model(image))
            pred_bin = (pred > 0.5).float()

            for b in range(image.shape[0]):
                mask_np = mask[b].cpu().squeeze(0).numpy()
                is_pos = mask_np.sum() > 0

                if is_pos and pos_shown >= pos_target:
                    continue
                if (not is_pos) and neg_shown >= neg_target:
                    continue

                img_np = denormalize(image[b].cpu()).permute(1, 2, 0).numpy()
                pred_np = pred_bin[b].cpu().squeeze(0).numpy()

                fig, axes = plt.subplots(1, 4, figsize=(18, 5))

                axes[0].imshow(img_np)
                axes[0].set_title(f"Immagine\n{file_names[b]}")
                axes[0].axis("off")

                axes[1].imshow(mask_np, cmap="gray")
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(pred_np, cmap="gray")
                axes[2].set_title("Predizione")
                axes[2].axis("off")

                axes[3].imshow(img_np)
                axes[3].imshow(pred_np, cmap="Reds", alpha=0.5)
                axes[3].set_title("Overlay")
                axes[3].axis("off")

                plt.tight_layout()

                tag = "positive" if is_pos else "negative"
                idx = pos_shown + 1 if is_pos else neg_shown + 1
                save_name = f"test_{tag}_{idx}_v2.png"
                plt.savefig(save_name, dpi=300, bbox_inches="tight")
                print("Salvato:", save_name)
                plt.show()
                plt.close()

                if is_pos:
                    pos_shown += 1
                else:
                    neg_shown += 1

                if pos_shown >= pos_target and neg_shown >= neg_target:
                    break

            if pos_shown >= pos_target and neg_shown >= neg_target:
                break

    print(f"Positivi: {pos_shown} | Negativi: {neg_shown}")


if __name__ == "__main__":
    freeze_support()
    main()
