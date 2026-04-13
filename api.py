from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import numpy as np
import cv2
from inference import predict
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
import base64

app = FastAPI()


def create_overlay(image, mask):

    image = np.array(image)

    # converter grayscale → RGB
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # ajustar tamanho da máscara
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # criar máscara vermelha
    red_mask = np.zeros_like(image)
    red_mask[:,:,0] = mask * 255

    # overlay
    overlay = cv2.addWeighted(image, 1.0, red_mask, 0.5, 0)

    return overlay

def generate_report(mask, raw_mask):

    lesion_pixels = int(mask.sum())

    total_pixels = mask.shape[0] * mask.shape[1]

    lesion_percentage = (lesion_pixels / total_pixels) * 100

    report = {
        "lesion_pixels": lesion_pixels,
        "total_pixels": total_pixels,
        "lesion_percentage": lesion_percentage,
        "max_probability": float(raw_mask.max()),
        "mean_probability": float(raw_mask.mean())
    }

    return report

def create_heatmap(image, raw_mask):

    image = np.array(image)

    # converter grayscale para RGB
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # redimensionar máscara
    raw_mask = cv2.resize(raw_mask, (image.shape[1], image.shape[0]))

    # converter probabilidade para 0-255
    heatmap = (raw_mask * 255).astype(np.uint8)

    # aplicar colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # combinar com imagem original
    overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)

    return overlay

@app.post("/segment")
async def segment(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("L")

    raw_mask = predict(image)

    print("Max mask:", raw_mask.max())
    print("Min mask:", raw_mask.min())

    cv2.imwrite("debug_mask.png", raw_mask * 255)

    mask = (raw_mask > 0.5).astype(np.uint8)

    report = generate_report(mask, raw_mask)

    # criar overlay de segmentação
    overlay = create_overlay(image, mask)

    # criar heatmap da rede
    heatmap = create_heatmap(image, raw_mask)

    # converter overlay
    _, buffer_overlay = cv2.imencode(".png", overlay)
    overlay_base64 = base64.b64encode(buffer_overlay.tobytes()).decode()

    # converter heatmap
    _, buffer_heatmap = cv2.imencode(".png", heatmap)
    heatmap_base64 = base64.b64encode(buffer_heatmap.tobytes()).decode()

    return JSONResponse({
        "report": report,
        "overlay": overlay_base64,
        "heatmap": heatmap_base64
    })