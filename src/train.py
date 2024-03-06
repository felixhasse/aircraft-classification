from datasets import FGVCDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from models import EfficientNet_B0
from torch.multiprocessing import freeze_support

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)

#mps_device = "cpu"

train_dataset = FGVCDataset("train", "family", transform=transform)
test_dataset = FGVCDataset("test", "family", transform=transform)
print("Dataset Loaded")

train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=True)

loss_fn = torch.nn.CrossEntropyLoss()

model = EfficientNet_B0(train_dataset.getNumClasses()).to(mps_device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
print("Starting Training...")

for epoch in range(10):
    model.train()
    i = 0
    for x, y in train_dataloader:
        x = x.to(mps_device)
        y = y.to(mps_device)
        x
        i += 1
        print("Training on batch", i, "of", len(train_dataloader))
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for x, y in test_dataloader:
            x = x.to(mps_device)
            y = y.to(mps_device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred, 1)
            y_class = torch.argmax(y, 1)
            print(predicted)
            print(y_class)
            total += y.size(0)
            correct += (predicted == y_class).sum().item()
        print(f"Epoch {epoch}: {correct/total}")

torch.save(model.state_dict(), "model.pth")