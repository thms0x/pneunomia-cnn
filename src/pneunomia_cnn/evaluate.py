import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from pneunomia_cnn.model import CNN
from pneunomia_cnn.dataset import XRayDataset


def evaluate():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    test_transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = XRayDataset(
        root_dir="data/chest_xray/test",
        transform=test_transform

    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False

    )

    model = CNN().to(device)
    state_dict = torch.load(
        "models/model.pt",
        map_location=device,
        weights_only=True)
    model.load_state_dict(state_dict=state_dict)
    model.eval()



    with torch.no_grad():
        correct_count = 0
        wrong_count = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).float()
            correct_count += (predictions == labels).sum().item()
            wrong_count += (predictions != labels).sum().item()
            total += labels.size(0)

        accuracy = correct_count / total
        print(f"Accuracy: {accuracy *100 :.2f}%")
        print(f"Correct: {correct_count}")
        print(f"Wrong: {wrong_count}")

if __name__ == "__main__":
    evaluate()