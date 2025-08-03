import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from urllib.request import urlretrieve
import os

def load_finetuned_resnet50(
    num_classes=17,
    diff_url="https://raw.githubusercontent.com/AmirHossienAfshar/cv-noise-denoise/master/saved_models/resnet50_finetune_diff.pth",
    local_diff_path="resnet50_finetune_diff.pth",
    base_model_path=None,
    device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load base
    if base_model_path and os.path.exists(base_model_path):
        print(f"[INFO] Loading base model from {base_model_path}")
        model = resnet50(weights=None)
        checkpoint = torch.load(base_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("[INFO] Loading base model from TorchVision")
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Adjust classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load diff
    if not os.path.exists(local_diff_path):
        print(f"[INFO] Downloading diff weights from {diff_url}")
        urlretrieve(diff_url, local_diff_path)

    diff_state = torch.load(local_diff_path, map_location=device)
    model_state = model.state_dict()
    model_state.update(diff_state)
    model.load_state_dict(model_state)

    model = model.to(device)
    model.eval()
    return model
