# ARQUIVO USADO PRA TRANSFORMAR DADOS NO FORMATO NIfTI PARA PNG

import os
# pyrefly: ignore [missing-import]
import nibabel as nib
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2

np.random.seed(42)

input_folder = "dataset_original"
output_images = "dataset/images"
output_masks = "dataset/masks"

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

index = 0

for patient in sorted(os.listdir(input_folder)):

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

            # Mantém todas as imagens com lesão e apenas 50% das sem lesão
            if np.max(msk) == 0:
                if np.random.rand() > 0.30:
                    continue

            if img.max() > img.min():
                img = (img - img.min()) / (img.max() - img.min())
            else:
                img = np.zeros_like(img)

            img = (img * 255).astype(np.uint8)
            msk = (msk > 0).astype(np.uint8) * 255

            img_name = f"img_{patient}_{i}.png"
            mask_name = f"mask_{patient}_{i}.png"

            cv2.imwrite(os.path.join(output_images, img_name), img)
            cv2.imwrite(os.path.join(output_masks, mask_name), msk)

            index += 1

print("Conversão finalizada:", index, "imagens")