import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import v2

import mlflow

from pneunomia_cnn.dataset import XRayDataset
from pneunomia_cnn.model import CNN

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3

def train():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("pneumonia-cnn")

    with mlflow.start_run():
        mlflow.log_params({
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "optimizer": "Adam",
        })
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        train_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5])
        ])

        val_transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5], std=[0.5])
        ])

        train_dataset = XRayDataset(
            "data/chest_xray/train",
            transform = train_transform
        )

        val_dataset = XRayDataset(
            "data/chest_xray/train",
            transform = val_transform
        )

        dataset_size = len(train_dataset)
        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(dataset_size, generator=generator)
        split = int(dataset_size * 0.85)
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
       
          

        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE,
            shuffle = True
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE,
            shuffle = False
        )

        model = CNN().to(device)

        criterion = nn.BCEWithLogitsLoss() # sigmoid + cross entropy loss

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=LEARNING_RATE
        )

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0

            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                optimizer.zero_grad()
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()


                print(
                    f"Epoch {epoch + 1}/{EPOCHS}, "
                    f"Loss: {loss:.5f}"
                )


            
            metrics = validate(model, val_loader, criterion, device)

            mlflow.log_metrics(
                {
                    "loss": metrics['loss'],
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1": metrics['f1'],
                },
                step=epoch,
            )

        torch.save(model.state_dict(), "models/model.pt")
        mlflow.log_artifact(
            "models/model.pt"
        )




def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    tp = 0
    fp = 0
    tn = 0 
    fn = 0


    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            tp += ((predictions == 1) & (labels == 1)).sum().item()
            fp += ((predictions == 1) & (labels == 0)).sum().item()
            tn += ((predictions == 0) & (labels == 0)).sum().item()
            fn += ((predictions == 0) & (labels == 1)).sum().item()

        accuracy = correct / total
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2*precision*recall) / (precision+recall)

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    


if __name__ == "__main__":
    train()

