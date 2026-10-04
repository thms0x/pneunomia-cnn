from pathlib import Path

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        class_map = {
            "NORMAL": 0,
            "PNEUMONIA": 1
        }

        for class_name, label in class_map.items():
            class_dir = self.root_dir / class_name

            for image_path in class_dir.glob("*.jpeg"):
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = read_image(str(image_path), mode=ImageReadMode.GRAY)

        if self.transform:
            image = self.transform(image)

        return image, label