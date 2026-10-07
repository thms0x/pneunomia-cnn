import torch 
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2

from pneunomia_cnn.model import CNN

def inference(path: str):

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

    image = read_image(
        path,
        mode = ImageReadMode.GRAY
    )

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        logits = model(image).unsqueeze(1)
        probability = torch.sigmoid(logits)

        prediction = (probability >= 0.5).item()

        if prediction:
            print(f"PNEUMONIA ({probability.item():.2%})")
        else:
            print(f"NORMAL ({1 - probability.item():.2%})")

if __name__ == "__main__":
    inference("data/examples/pneunomia.jpg")
    inference("data/examples/normal.jpg")

