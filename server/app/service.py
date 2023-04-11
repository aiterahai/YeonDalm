"""
file name: service.py

create time: 2023-04-05 10:15
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
import io

from fastapi import APIRouter, status, UploadFile
from fastapi.param_functions import File
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import os
from PIL import Image

service_router = APIRouter()

model = EfficientNet.from_pretrained('efficientnet-b0')
num_classes = len(os.listdir('../ai/data'))
model._fc = nn.Linear(model._fc.in_features, num_classes)

checkpoint = torch.load('../ai/model/150model.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

classes = os.listdir('../ai/data')

@service_router.post("/pred", status_code=status.HTTP_200_OK)
async def predict_image(image : UploadFile = File(...)):
    contents = await image.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.topk(output, k=3, dim=1)
    return [classes[i] for i in predicted[0]]