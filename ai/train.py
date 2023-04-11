"""
file name: train.py

create time: 2023-04-03 15:01
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import os

model = EfficientNet.from_pretrained('efficientnet-b0')
num_classes = len(os.listdir('data'))
model._fc = nn.Linear(model._fc.in_features, num_classes)

train_dataset = ImageFolder('data', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

num_epochs = 150
path = "model/"

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in tqdm(range(1, num_epochs + 1)):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : loss,
        }, f"{path}{epoch}model.pt")

    print(f"epoch : {epoch}, loss : {loss}")