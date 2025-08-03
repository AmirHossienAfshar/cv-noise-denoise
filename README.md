# cv-noise-denoise
Noise classification and denoising for agricultural images using CNNs

---

### Fine-Tuned ResNet50 Inference Helper

This project provides a lightweight way to use a fine-tuned ResNet50 without uploading the entire model checkpoint. Only the *fine-tuned differences* (`resnet50_finetune_diff.pth`) are stored in this repository, while the base model weights are pulled from TorchVision (or optionally loaded from your own local checkpoint).

#### How It Works
- **Base model:** Standard `ResNet50_Weights.IMAGENET1K_V1` from TorchVision (or your own `.pth`).
- **Diff file:** Contains only the parameters that were fine-tuned (much smaller file).
- At runtime, the script:
  1. Loads the base model.
  2. Adjusts the final classification layer for your dataset (17 classes).
  3. Downloads and applies the diff weights.
  4. Returns a ready-to-use model for inference.


#### Quick Start on inference

##### 1. Open the inference notebook
```bash
jupyter notebook inference_helper.ipynb
```

##### 2. Use the Helper Function

The key function is `load_finetuned_resnet50`, which will handle everything automatically:

```python
from inference_helper import load_finetuned_resnet50

# Load fine-tuned ResNet50 (downloads diff if not present)
model = load_finetuned_resnet50()
model.eval()
```


##### Optional: Provide Your Own Base Model

If you already have a base ResNet50 `.pth` (e.g., custom pretraining):

```python
model = load_finetuned_resnet50(
    base_model_path="path/to/your/base_model.pth"
)
```

This bypasses downloading TorchVision weights.

---