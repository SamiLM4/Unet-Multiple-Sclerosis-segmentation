import requests
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import io
import base64

# URL da API
URL = "http://127.0.0.1:8000/segment"

# criar janela invisível
root = tk.Tk()
root.withdraw()

# selecionar arquivo
file_path = filedialog.askopenfilename(
    title="Selecione uma imagem MRI",
    filetypes=[("Imagens", "*.png *.jpg *.jpeg")]
)

if not file_path:
    print("Nenhum arquivo selecionado.")
    exit()

print("Enviando imagem para a IA...")

# enviar imagem
with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(URL, files=files)

# verificar resposta
if response.status_code == 200:

    data = response.json()

    report = data["report"]
    overlay_base64 = data["overlay"]
    heatmap_base64 = data["heatmap"]

    # mostrar relatório
    print("\n===== RELATÓRIO DA IA =====")

    print("Pixels de lesão:", report["lesion_pixels"])
    print("Total de pixels:", report["total_pixels"])
    print("Área de lesão (%):", round(report["lesion_percentage"], 4))
    print("Probabilidade máxima:", round(report["max_probability"], 4))
    print("Probabilidade média:", round(report["mean_probability"], 4))

    # decodificar imagem
    overlay_bytes = base64.b64decode(overlay_base64)
    heatmap_bytes = base64.b64decode(heatmap_base64)

    overlay_image = Image.open(io.BytesIO(overlay_bytes))
    heatmap_image = Image.open(io.BytesIO(heatmap_bytes))

    # salvar resultado
    overlay_image.save("resultado_overlay.png")
    heatmap_image.save("resultado_heatmap.png")

    print("\nImagens salvas")

    # mostrar imagem
    overlay_image.show()
    heatmap_image.show()

else:
    print("Erro:", response.status_code)