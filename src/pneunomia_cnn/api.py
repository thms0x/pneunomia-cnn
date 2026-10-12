import torch 
from fastapi import FastAPI, UploadFile, HTTPException
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

from pneunomia_cnn.model import CNN

app = FastAPI(title="Pneunomia CNN API")

device = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

transform = v2.Compose([
v2.Resize((224, 224)),
v2.ToDtype(torch.float32, scale=True),
v2.Normalize(mean=[0.5], std=[0.5]),
])

model = CNN().to(device)

state_dict = torch.load(
"models/model.pt",
map_location=device, 
weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile):
    if file.content_type not in {
        "image/jpeg",
        "image/png",
    }:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported",
        )

    contents = await file.read()

    encoded = torch.frombuffer(
        bytearray(contents),
        dtype=torch.uint8,
    )

    image = decode_image(
        encoded,
        mode=ImageReadMode.GRAY,
    )

    image = transform(image)

    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(image).squeeze()
        probability = torch.sigmoid(logit).item()

    prediction = (
        "PNEUMONIA"
        if probability >= 0.5
        else "NORMAL"
    )

    return {
        "prediction": prediction,
        "pneumonia_score": probability,
    }