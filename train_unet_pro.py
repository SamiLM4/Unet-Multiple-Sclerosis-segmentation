import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm
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
# TRANSFORMS (CORREÇÃO CRÍTICA)
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

    probs = probs.view(-1)
    targets = targets.view(-1)

    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum()

    return (2. * intersection) / (union + 1e-6)


############################################
# SPLIT DATASET
############################################

def split_dataset(dataset, val_ratio=0.2):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


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

    dataset = MRIDataset(
        image_dir="dataset/images",
        mask_dir="dataset/masks",
        img_transform=img_transform,
        mask_transform=mask_transform
    )

    train_set, val_set = split_dataset(dataset)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    model = UNet().to(device)

    bce = nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()

    def criterion(pred, target):
        return bce(pred, target) + dice_loss(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model_path = "unet_mri_model.pth"

    if os.path.exists(model_path):
        print("Carregando modelo existente...")
        model.load_state_dict(torch.load(model_path, map_location=device))

    best_val_dice = 0

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_dice = validate_epoch(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), model_path)
            print("🔥 Melhor modelo salvo!")

    print("\nTreinamento finalizado!")


############################################
# RUN
############################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    main(args.epochs)