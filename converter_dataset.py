# ARQUIVO USADO PRA TRANSFORMAR DADOS NO FORMATO NIfTI PARA PNG

import os
import nibabel as nib
import numpy as np
import cv2

input_folder = "dataset_original"
output_images = "dataset/images"
output_masks = "dataset/masks"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

index = 0

for patient in os.listdir(input_folder):

    patient_path = os.path.join(input_folder, patient)

    if not os.path.isdir(patient_path):
        continue

    flair_path = None
    mask_path = None

    for file in os.listdir(patient_path):

        if "Flair.nii" in file and "LesionSeg" not in file:
            flair_path = os.path.join(patient_path, file)

        if "LesionSeg-Flair" in file:
            mask_path = os.path.join(patient_path, file)

    if flair_path and mask_path:

        print("Processando:", patient)

        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        slices = flair.shape[2]

        for i in range(slices):

            img = flair[:,:,i]
            msk = mask[:,:,i]

            if np.max(msk) == 0:
                continue

            img = (img - img.min())/(img.max()-img.min())
            img = (img*255).astype(np.uint8)

            msk = (msk > 0).astype(np.uint8) * 255

            cv2.imwrite(f"{output_images}/img_{index}.png", img)
            cv2.imwrite(f"{output_masks}/mask_{index}.png", msk)

            index += 1

print("Conversão finalizada:", index, "imagens")