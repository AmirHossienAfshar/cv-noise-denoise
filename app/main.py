from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_loader import load_finetuned_resnet50
from app.utils import preprocess_image
import torch

app = FastAPI(title="Leaf Noise Classifier")

static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

model = load_finetuned_resnet50()

labels = [
    "Gaussian", "Salt & Pepper", "Poisson", "Speckle", "Uniform",
    "Gaussian + Salt & Pepper", "Gaussian + Poisson", "Gaussian + Speckle",
    "Gaussian + Uniform", "Salt & Pepper + Speckle", "Salt & Pepper + Uniform",
    "Poisson + Speckle", "Poisson + Uniform", "Speckle + Uniform",
    "gaussian + salt & pepper + speckle", "gaussian + poisson + uniform", "salt & pepper + speckle + uniform"
]

@app.get("/", response_class=HTMLResponse)
def root():
    static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "index.html")
    with open(static_path) as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = outputs.argmax(dim=1).item()
        pred_label = labels[pred_idx]

    return JSONResponse({"predicted_class": pred_label})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)