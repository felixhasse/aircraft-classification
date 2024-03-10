from torch.utils.data import Dataset
import torch
from typing import Union
import PIL.Image

DatasetType = Union[str, "train", "val", "test"]
LabelType = Union[str, "family", "manufacturer", "variant"]


class FGVCDataset(Dataset):
    def __init__(self, data: DatasetType, label: LabelType, transform=None):

        match label:
            case "family":
                classes_path = f"./../data/families.txt"
            case "manufacturer":
                classes_path = f"./../data/manufacturers.txt"
            case "variant":
                classes_path = f"./../data/variants.txt"

        with open(classes_path, "r") as f:
            lines = f.readlines()
            self.classes = [line.strip() for line in lines]

        file_path = f"./../data/images_{label}_{data}.txt"

        with open(file_path, "r") as f:
            lines = f.readlines()
            data = []
            for line in lines:
                data.append(
                    f"./../data/images/{line.split(" ")[0]}.jpg")
            self.data = data
            labels = [line.split(" ", 1)[1].strip() for line in lines]
            self.labels = list(map(lambda x: torch.tensor(
                [1 if x == c else 0 for c in self.classes], dtype=torch.float32), labels))
            
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = PIL.Image.open(self.data[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def getLabeledClasses(self, label: torch.Tensor):
        return [(self.classes[i], label[i].item()) for i in range(len(label))]
    
    def getClassFromLabel(self, label: torch.Tensor):
        return self.classes[label.argmax().item()]

    def getNumClasses(self):
        return len(self.classes)
