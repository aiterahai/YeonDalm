"""
file name: crop.py

create time: 2023-05-01 13:44
author: Tera Ha
e-mail: terra2007@naver.com
github: https://github.com/terra2007
"""
import os
import cv2
import numpy as np
from multiprocessing import Pool


def process_image(image_path, label_output_dir):
    # OpenCV를 사용하여 이미지를 로드합니다.
    img_array = np.fromfile(image_path, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # OpenCV의 얼굴 인식 알고리즘을 사용하여 얼굴을 탐지합니다.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # 각 얼굴을 crop하여 저장합니다.
    for i, (x, y, w, h) in enumerate(faces):
        cropped_image = image[y:y+h, x:x+w]
        output_file = os.path.join(label_output_dir, f"{i}_{os.path.basename(image_path)}")
        ret, img_arr = cv2.imencode(".jpg", cropped_image)
        if ret:
            with open(output_file, mode='wb') as f:
                img_arr.tofile(f)


if __name__ == '__main__':
    # 입력 이미지와 라벨 디렉토리를 지정합니다.
    image_dir = 'data/'
    label_dir = 'data/'
    output_dir = 'cropped/'

    # 라벨 디렉토리 내의 모든 라벨을 가져옵니다.
    labels = os.listdir(label_dir)

    # 출력 디렉토리가 없는 경우 생성합니다.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 멀티프로세싱 풀을 생성합니다.
    pool = Pool()

    # 모든 라벨에 대해 이미지를 crop합니다.
    for label in labels:
        # 라벨 디렉토리에서 해당 라벨의 이미지 파일을 가져옵니다.
        label_image_dir = os.path.join(image_dir, label)
        image_paths = [os.path.join(label_image_dir, f) for f in os.listdir(label_image_dir)]

        # 출력 디렉토리에 해당 라벨의 이름으로 서브디렉토리를 생성합니다.
        label_output_dir = os.path.join(output_dir, label)
        if not os.path.exists(label_output_dir):
            os.makedirs(label_output_dir)

        # 멀티프로세싱으로 crop 작업을 수행합니다.
        pool.starmap(process_image, [(p, label_output_dir) for p in image_paths])
    # 멀티프로세싱 풀을 종료합니다.
    pool.close()
    pool.join()
