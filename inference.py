# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from torchvision import transforms
# pyrefly: ignore [missing-import]
from model import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
checkpoint = torch.load("unet_mri_model.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])


def predict(image: Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    return output.squeeze().cpu().numpy()  # probabilidade, sem threshold