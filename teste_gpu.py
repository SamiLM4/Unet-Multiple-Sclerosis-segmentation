# pyrefly: ignore [missing-import]
import torch

print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("Versão CUDA do PyTorch:", torch.version.cuda)
print("Quantidade de GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Nenhuma GPU CUDA encontrada.")