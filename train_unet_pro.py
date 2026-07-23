# FAZ A MESMA COISA QUE O NOTEBOOK, PREFIRA EXECUTAR ESSE ARQUIVO PARA TREINAR O MODELO DE IA 

import os
import random
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader, Subset
# pyrefly: ignore [missing-import]
import torchvision.transforms.functional as TF
# pyrefly: ignore [missing-import]
from torchvision import transforms
# pyrefly: ignore [missing-import]
from torchvision.transforms import InterpolationMode
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from tqdm import tqdm
# pyrefly: ignore [missing-import]
import argparse


############################################
# DATASET
############################################

class MRIDataset(Dataset):

    def __init__(self, image_dir, mask_dir, img_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace("img", "mask"))

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.img_transform:
            image = self.img_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        # binariza máscara corretamente
        mask = (mask > 0.5).float()

        return image, mask


############################################
# TRANSFORMS
############################################

img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
    transforms.ToTensor()
])


############################################
# DATASET COM DATA AUGMENTATION
############################################

class MRIDatasetAug(Dataset):
    """
    Dataset de treino com augmentation sincronizada entre imagem e máscara.
    Augmentações geométricas (flip, rotação) são aplicadas igualmente nos dois.
    Jitter de brilho/contraste é aplicado SOMENTE na imagem.
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace("img", "mask"))

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # --- Redimensionar ---
        image = TF.resize(image, [256, 256])
        mask = TF.resize(mask, [256, 256], interpolation=TF.InterpolationMode.NEAREST)

        # --- Augmentações geométricas sincronizadas ---
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

        # --- Jitter somente na imagem (não faz sentido na máscara binária) ---
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
        if random.random() > 0.5:
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))

        # --- Converter para tensor ---
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Binariza máscara
        mask = (mask > 0.5).float()

        return image, mask


############################################
# UNET
############################################

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        b = self.bottleneck(p2)

        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1)
        u1 = self.conv1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1)
        u2 = self.conv2(u2)

        return self.out(u2)


############################################
# LOSS FUNCTIONS
############################################

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def dice_score(logits, targets, threshold=0.5):

    probs = torch.sigmoid(logits)
    probs = (probs > threshold).float()

    intersection = (probs * targets).sum(dim=(1,2,3))
    union = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))

    dice = (2 * intersection + 1e-6) / (union + 1e-6)

    return dice.mean()


############################################
# SPLIT DATASET
############################################

def split_dataset(dataset, val_ratio=0.2, seed=42):

    np.random.seed(seed)

    # Descobre quais pacientes existem
    patients = sorted({
        img.split("_")[1]
        for img in dataset.images
    })

    np.random.shuffle(patients)

    split = int(len(patients) * (1 - val_ratio))

    train_patients = set(patients[:split])
    val_patients = set(patients[split:])

    train_indices = []
    val_indices = []

    for idx, img in enumerate(dataset.images):

        patient = img.split("_")[1]

        if patient in train_patients:
            train_indices.append(idx)
        else:
            val_indices.append(idx)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


############################################
# TRAIN
############################################

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0
    total_dice = 0

    loop = tqdm(loader, desc="Train")

    for images, masks in loop:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice = dice_score(outputs, masks)

        total_loss += loss.item()
        total_dice += dice.item()

        loop.set_postfix(loss=loss.item(), dice=dice.item())

    return total_loss / len(loader), total_dice / len(loader)


############################################
# VALIDATION
############################################

def validate_epoch(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    total_dice = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)
            dice = dice_score(outputs, masks)

            total_loss += loss.item()
            total_dice += dice.item()

    return total_loss / len(loader), total_dice / len(loader)


############################################
# MAIN
############################################

def main(epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset base (sem aug) — usado para split e para validação
    dataset = MRIDataset(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        img_transform=img_transform,
        mask_transform=mask_transform
    )

    train_set, val_set = split_dataset(dataset)

    # Substituir o subset de treino por dataset com augmentation
    train_aug = MRIDatasetAug(
        image_dir="dataset/images",
        mask_dir="dataset/masks"
    )
    # Manter apenas os índices do split de treino
    train_set_aug = Subset(train_aug, train_set.indices)

    train_loader = DataLoader(train_set_aug, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    model = UNet().to(device)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=device))
    dice_loss = DiceLoss()

    def criterion(pred, target):
        return bce(pred, target) + dice_loss(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model_path = "unet_mri_model.pth"
    history_path = "training_history.csv"
    start_epoch = 0
    resuming = os.path.exists(model_path)

    best_val_dice = 0.0

    if resuming:
        print("Carregando modelo existente...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_dice = checkpoint.get('best_val_dice', 0.0)

            write_mode = 'a' if os.path.exists(history_path) else 'w'
            if write_mode == 'a':
                try:
                    with open(history_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) > 1:
                            last_line = lines[-1].strip().split(',')
                            start_epoch = int(last_line[0])
                except Exception:
                    pass
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            write_mode = 'w'
    else:
        write_mode = 'w'

    if write_mode == 'w':
        with open(history_path, 'w', encoding='utf-8') as f:
            f.write("epoch,train_loss,train_dice,val_loss,val_dice\n")
    

    for epoch in range(epochs):
        current_epoch = start_epoch + epoch + 1
        print(f"\nEpoch {current_epoch}/{start_epoch + epochs}")

        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_dice = validate_epoch(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")

        # Salva o histórico no arquivo CSV
        with open(history_path, 'a', encoding='utf-8') as f:
            f.write(f"{current_epoch},{train_loss},{train_dice},{val_loss},{val_dice}\n")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_dice': best_val_dice
            }, model_path)
            print("🔥 Melhor modelo salvou!")

    print("\nTreinamento finalizado!")


############################################
# RUN python train_unet_pro.py --epochs 20
############################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    main(args.epochs)