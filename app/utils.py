from torchvision import transforms
from PIL import Image
import io

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_image(file):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return transform(image).unsqueeze(0)  # [1, 3, 224, 224]
