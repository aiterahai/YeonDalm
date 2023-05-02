"""
file name: service.py

create time: 2023-04-05 10:15
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
import io

from fastapi import APIRouter, UploadFile
from fastapi.param_functions import File
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as transforms
import os
from PIL import Image
import cv2
import numpy as np

service_router = APIRouter()

model = EfficientNet.from_name('efficientnet-b0')
num_classes = len(os.listdir('../ai/data'))
model._fc = nn.Linear(model._fc.in_features, num_classes)

checkpoint = torch.load('../ai/model/best_model.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

classes = os.listdir('../ai/data')

def crop_image(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 첫 번째 얼굴 영역만 자릅니다.
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        cropped_img = img[y:y+h, x:x+w]
        return [cropped_img]
    else:
        return []

@service_router.post("/pred")
async def predict_image(image: UploadFile = File(...)):
    # 이미지 파일을 읽어옵니다.
    contents = await image.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), -1)

    # 이미지를 자립니다.
    cropped_imgs = crop_image(img)

    # 자른 이미지를 분류합니다.
    results = []
    for cropped_img in cropped_imgs:
        image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            top3_probs, top3_idxs = torch.topk(output, k=3, dim=1)

        result = [classes[i] for i in top3_idxs[0]]
        results.append(result)

    if len(results) > 0:
        return results[0]
    else:
        return []


    return results