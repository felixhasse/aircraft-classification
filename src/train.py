from datasets import FGVCDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from models import EfficientNet_B0
from models import EfficientNet_V2_S
from models import ViT_L_16
from torch.multiprocessing import freeze_support
from torchvision.models import EfficientNet_V2_S_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms._presets import ImageClassification

if __name__ == '__main__':
    freeze_support()
    transform = ImageClassification(
        crop_size=[128],
        resize_size=[128],
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        interpolation=InterpolationMode.BILINEAR
    )

    batch_size = 4
    lr = 0.00003

    transform = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()


    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)

    #mps_device = "cpu"

    train_dataset = FGVCDataset("train", "family", transform=transform)
    test_dataset = FGVCDataset("test", "family", transform=transform)
    print("Dataset Loaded")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    model = EfficientNet_V2_S(train_dataset.getNumClasses()).to(mps_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Starting Training...")

    min_loss = float("inf")
    non_improved = 0


    for epoch in range(10):
        model.train()
        i = 0
        total_loss = 0
        for x, y in train_dataloader:
            print(i)
            x = x.to(mps_device)
            y = y.to(mps_device)
            x
            i += 1
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(f"Training Loss: {total_loss/len(train_dataloader)}")
        model.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            total_loss = 0
            for x, y in test_dataloader:
                x = x.to(mps_device)
                y = y.to(mps_device)
                y_pred = model(x)
                total_loss += loss_fn(y_pred, y)
                _, predicted = torch.max(y_pred, 1)
                y_class = torch.argmax(y, 1)
                total += y.size(0)
                correct += (predicted == y_class).sum().item()
            print(f"Epoch {epoch}: {correct/total}")
            print(f"Evaluation Loss: {total_loss/len(test_dataloader)}")
            if total_loss < min_loss:
                min_loss = total_loss
                torch.save(model.state_dict(), f"models/{lr}_{batch_size}_{epoch}/.pth")
                non_improved = 0
            else:
                non_improved += 1
                if non_improved > 3:
                    break