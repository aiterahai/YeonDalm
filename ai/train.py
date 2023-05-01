import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 경로와 클래스 수
data_dir = 'cropped/'
num_classes = len(os.listdir(data_dir))

# 모델 정의
model = EfficientNet.from_name('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, num_classes)

# 모델을 GPU로 옮김
model = model.to(device)

# 데이터셋 및 데이터로더 정의
dataset = ImageFolder(data_dir, transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
]))

# 데이터셋을 train, valid로 나누기
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

# 학습 파라미터
num_epochs = 150
patience = 10  # early stopping 조기 종료를 위한 기다리는 epoch 수
best_valid_loss = float('inf')  # 초기값을 무한대로 설정하여 항상 첫 번째 epoch에서 저장되도록 함

# 손실 함수와 optimizer 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

path = 'model/'

# early stopping을 적용한 학습 루프
for epoch in tqdm(range(1, num_epochs + 1)):
    train_loss = 0.0
    valid_loss = 0.0

    # train 데이터셋으로 학습
    model.train()
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    # valid 데이터셋으로 평가
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)

    # 평균 손실을 계산
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)

    # epoch 마다 loss 출력
    print(f"Epoch {epoch}: train_loss = {train_loss}, valid_loss = {valid_loss}")

    # 현재까지 가장 좋은 valid_loss가 갱신된 경우 모델 저장
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        early_stop_count = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'valid_loss': valid_loss
        }, f"{path}best_model.pt")
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    # Save model every 10 epochs
    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'valid_loss': valid_loss
        }, f"{path}{epoch}model.pt")

    print(f"epoch : {epoch}, train_loss : {loss}, val_loss : {valid_loss}")