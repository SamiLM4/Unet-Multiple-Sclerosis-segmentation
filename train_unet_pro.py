# train_unet_regularized.py
#
# U-Net para segmentação de lesões de Esclerose Múltipla
# Versão regularizada para reduzir overfitting

import os
import random
import argparse

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



############################################
# DATASET VALIDAÇÃO
############################################

class MRIDataset(Dataset):

    def __init__(
        self,
        image_dir,
        mask_dir,
        img_transform=None,
        mask_transform=None
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(
            os.listdir(image_dir)
        )

        self.img_transform = img_transform
        self.mask_transform = mask_transform


    def __len__(self):

        return len(self.images)



    def __getitem__(self,index):

        img_name = self.images[index]


        img_path = os.path.join(
            self.image_dir,
            img_name
        )

        mask_path = os.path.join(
            self.mask_dir,
            img_name.replace(
                "img",
                "mask"
            )
        )


        image = Image.open(
            img_path
        ).convert("L")


        mask = Image.open(
            mask_path
        ).convert("L")



        if self.img_transform:
            image = self.img_transform(image)


        if self.mask_transform:
            mask = self.mask_transform(mask)



        mask = (
            mask > 0.5
        ).float()



        return image, mask





############################################
# TRANSFORMS
############################################


img_transform = transforms.Compose([

    transforms.Resize(
        (256,256)
    ),

    transforms.ToTensor()

])



mask_transform = transforms.Compose([

    transforms.Resize(
        (256,256),
        interpolation=InterpolationMode.NEAREST
    ),

    transforms.ToTensor()

])






############################################
# DATASET TREINO COM AUGMENTATION
############################################


class MRIDatasetAug(Dataset):


    def __init__(
        self,
        image_dir,
        mask_dir
    ):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(
            os.listdir(image_dir)
        )



    def __len__(self):

        return len(self.images)




    def __getitem__(self,index):


        img_name = self.images[index]


        img_path = os.path.join(
            self.image_dir,
            img_name
        )


        mask_path = os.path.join(
            self.mask_dir,
            img_name.replace(
                "img",
                "mask"
            )
        )



        image = Image.open(
            img_path
        ).convert("L")


        mask = Image.open(
            mask_path
        ).convert("L")



        image = TF.resize(
            image,
            [256,256]
        )


        mask = TF.resize(
            mask,
            [256,256],
            interpolation=InterpolationMode.NEAREST
        )




        # Flip horizontal

        if random.random() > 0.5:

            image = TF.hflip(image)
            mask = TF.hflip(mask)




        # Flip vertical

        if random.random() > 0.5:

            image = TF.vflip(image)
            mask = TF.vflip(mask)




        # Rotação

        if random.random() > 0.5:

            angle = random.uniform(
                -15,
                15
            )

            image = TF.rotate(
                image,
                angle
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST
            )




        # Brilho

        if random.random() > 0.5:

            image = TF.adjust_brightness(
                image,
                random.uniform(
                    0.8,
                    1.2
                )
            )




        # Contraste

        if random.random() > 0.5:

            image = TF.adjust_contrast(
                image,
                random.uniform(
                    0.8,
                    1.2
                )
            )



        image = TF.to_tensor(image)

        mask = TF.to_tensor(mask)




        # Ruído de RM somente treino

        if random.random() > 0.5:

            noise = torch.randn_like(image) * 0.05

            image = image + noise

            image = torch.clamp(
                image,
                0,
                1
            )



        mask = (
            mask > 0.5
        ).float()



        return image, mask

############################################
# UNET REGULARIZADA
############################################


class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()


        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),


            nn.Dropout2d(
                0.2
            ),


            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),


            nn.BatchNorm2d(
                out_channels
            ),


            nn.ReLU(
                inplace=True
            )
        )



    def forward(self,x):

        return self.conv(x)




class UNet(nn.Module):


    def __init__(self):

        super().__init__()



        # Encoder

        self.down1 = DoubleConv(
            1,
            32
        )

        self.pool1 = nn.MaxPool2d(
            2
        )



        self.down2 = DoubleConv(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(
            2
        )



        # Bottleneck

        self.bottleneck = DoubleConv(
            64,
            128
        )




        # Decoder


        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )


        self.conv1 = DoubleConv(
            128,
            64
        )



        self.up2 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )


        self.conv2 = DoubleConv(
            64,
            32
        )



        self.out = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )




    def forward(self,x):


        d1 = self.down1(x)

        p1 = self.pool1(d1)



        d2 = self.down2(p1)

        p2 = self.pool2(d2)



        b = self.bottleneck(p2)




        u1 = self.up1(b)


        u1 = torch.cat(
            [
                u1,
                d2
            ],
            dim=1
        )


        u1 = self.conv1(u1)




        u2 = self.up2(u1)


        u2 = torch.cat(
            [
                u2,
                d1
            ],
            dim=1
        )


        u2 = self.conv2(u2)



        return self.out(u2)







############################################
# LOSS FUNCTIONS
############################################


class DiceLoss(nn.Module):


    def __init__(
        self,
        smooth=1e-6
    ):

        super().__init__()

        self.smooth = smooth




    def forward(
        self,
        logits,
        targets
    ):


        probs = torch.sigmoid(
            logits
        )


        probs = probs.view(-1)

        targets = targets.view(-1)



        intersection = (
            probs * targets
        ).sum()



        dice = (

            2.0 * intersection
            + self.smooth

        ) / (

            probs.sum()
            + targets.sum()
            + self.smooth

        )


        return 1 - dice






def dice_score(
    logits,
    targets,
    threshold=0.5
):


    probs = torch.sigmoid(
        logits
    )


    probs = (
        probs > threshold
    ).float()



    intersection = (
        probs * targets
    ).sum(
        dim=(1,2,3)
    )



    union = (

        probs.sum(
            dim=(1,2,3)
        )

        +

        targets.sum(
            dim=(1,2,3)
        )

    )



    dice = (

        2 * intersection
        + 1e-6

    ) / (

        union
        + 1e-6

    )



    return dice.mean()







############################################
# SPLIT POR PACIENTE
############################################


def split_dataset(
    dataset,
    val_ratio=0.2,
    seed=42
):


    np.random.seed(
        seed
    )



    patients = sorted(
        {
            img.split("_")[1]
            for img in dataset.images
        }
    )



    np.random.shuffle(
        patients
    )



    split = int(
        len(patients)
        *
        (1-val_ratio)
    )



    train_patients = set(
        patients[:split]
    )


    val_patients = set(
        patients[split:]
    )



    train_indices = []

    val_indices = []




    for idx,img in enumerate(
        dataset.images
    ):


        patient = img.split("_")[1]



        if patient in train_patients:

            train_indices.append(
                idx
            )

        else:

            val_indices.append(
                idx
            )




    train_dataset = Subset(
        dataset,
        train_indices
    )


    val_dataset = Subset(
        dataset,
        val_indices
    )



    return (
        train_dataset,
        val_dataset
    )

############################################
# TRAIN
############################################


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()


    total_loss = 0
    total_dice = 0



    loop = tqdm(
        loader,
        desc="Train"
    )



    for images, masks in loop:


        images = images.to(device)

        masks = masks.to(device)



        outputs = model(images)



        loss = criterion(
            outputs,
            masks
        )



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        dice = dice_score(
            outputs,
            masks
        )



        total_loss += loss.item()

        total_dice += dice.item()



        loop.set_postfix(

            loss=loss.item(),

            dice=dice.item()

        )




    return (

        total_loss / len(loader),

        total_dice / len(loader)

    )







############################################
# VALIDATION
############################################


def validate_epoch(
    model,
    loader,
    criterion,
    device
):


    model.eval()



    total_loss = 0

    total_dice = 0




    with torch.no_grad():


        for images,masks in loader:


            images = images.to(device)

            masks = masks.to(device)



            outputs = model(images)



            loss = criterion(
                outputs,
                masks
            )


            dice = dice_score(
                outputs,
                masks
            )



            total_loss += loss.item()

            total_dice += dice.item()




    return (

        total_loss / len(loader),

        total_dice / len(loader)

    )








############################################
# MAIN
############################################


def main(epochs):


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )



    print(
        "Usando dispositivo:",
        device
    )





    ########################################
    # DATASET
    ########################################


    dataset = MRIDataset(

        image_dir="dataset/images",

        mask_dir="dataset/masks",

        img_transform=img_transform,

        mask_transform=mask_transform

    )




    train_set, val_set = split_dataset(
        dataset
    )



    train_aug = MRIDatasetAug(

        image_dir="dataset/images",

        mask_dir="dataset/masks"

    )



    train_set_aug = Subset(

        train_aug,

        train_set.indices

    )




    train_loader = DataLoader(

        train_set_aug,

        batch_size=4,

        shuffle=True

    )



    val_loader = DataLoader(

        val_set,

        batch_size=4,

        shuffle=False

    )





    ########################################
    # MODEL
    ########################################


    model = UNet().to(device)





    ########################################
    # LOSS
    ########################################


    bce = nn.BCEWithLogitsLoss(

        pos_weight=torch.tensor(

            [10.0],

            device=device

        )

    )



    dice_loss = DiceLoss()




    def criterion(
        pred,
        target
    ):

        return (

            bce(pred,target)

            +

            dice_loss(pred,target)

        )







    ########################################
    # OPTIMIZER
    ########################################


    optimizer = optim.Adam(

        model.parameters(),

        lr=1e-4

    )




    scheduler = optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        patience=4,

        factor=0.5

    )








    ########################################
    # CHECKPOINT
    ########################################


    model_path = (

        "unet_mri_regularized.pth"

    )


    history_path = (

        "training_history_regularized.csv"

    )



    best_val_dice = 0.0


    start_epoch = 0




    # NÃO carregar modelo antigo
    # Essa arquitetura é diferente da anterior





    ########################################
    # HISTÓRICO
    ########################################


    if not os.path.exists(history_path):

        with open(
            history_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(

                "epoch,train_loss,train_dice,val_loss,val_dice\n"

            )







    ########################################
    # EARLY STOPPING
    ########################################


    patience = 8

    counter = 0






    ########################################
    # LOOP TREINO
    ########################################


    for epoch in range(
        epochs
    ):



        current_epoch = (

            start_epoch

            +

            epoch

            +

            1

        )



        print(

            f"\nEpoch {current_epoch}/{epochs}"

        )




        train_loss, train_dice = train_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device

        )




        val_loss, val_dice = validate_epoch(

            model,

            val_loader,

            criterion,

            device

        )




        scheduler.step(
            val_dice
        )




        print(

            f"Train Loss: {train_loss:.4f} | "
            f"Train Dice: {train_dice:.4f}"

        )


        print(

            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"

        )





        with open(

            history_path,

            "a",

            encoding="utf-8"

        ) as f:


            f.write(

                f"{current_epoch},"
                f"{train_loss},"
                f"{train_dice},"
                f"{val_loss},"
                f"{val_dice}\n"

            )







        ####################################
        # MELHOR MODELO
        ####################################


        if val_dice > best_val_dice:


            best_val_dice = val_dice

            counter = 0



            torch.save(

                {

                    "epoch":
                    current_epoch,


                    "model_state_dict":
                    model.state_dict(),


                    "optimizer_state_dict":
                    optimizer.state_dict(),


                    "best_val_dice":
                    best_val_dice

                },

                model_path

            )


            print(
                "🔥 Melhor modelo salvo!"
            )




        else:


            counter += 1


            print(

                f"Sem melhoria: "
                f"{counter}/{patience}"

            )



            if counter >= patience:


                print(

                    "⛔ Early stopping ativado"

                )


                break







    print(
        "\nTreinamento finalizado!"
    )

    print(
        "Melhor Dice:",
        best_val_dice
    )







############################################
# EXECUÇÃO
############################################


if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--epochs",

        type=int,

        default=50

    )



    args = parser.parse_args()



    main(
        args.epochs
    )