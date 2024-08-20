import torch
from torchvision import datasets, transforms
import cv2
import numpy as np
import pandas as pd
import os
import random

DATAROOT = '/home/dolphin/choi_ws/SatLaneDet_2024/dataset'


def generate_dataset_main():
    src_dataset = prepare_original_dataset(DATAROOT)
    generate_image_dot_dataset(src_dataset, os.path.join(DATAROOT, 'train'), (0, 2_000))
    generate_image_dot_dataset(src_dataset, os.path.join(DATAROOT, 'test'), (2_000, 3_000))


def prepare_original_dataset(data_path):
    # CIFAR-10 데이터셋 로드 및 전처리
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    cifar10 = datasets.CIFAR10(root=data_path + '/cifar10', train=True, download=True, transform=transform)
    return cifar10


def generate_image_dot_dataset(src_dataset, data_path, index_range):
    # 저장 경로 설정
    image_path = os.path.join(data_path, 'image')
    label_path = os.path.join(data_path, 'label')
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)
    # 색상 정의
    colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0)
    }
    # 데이터프레임 초기화
    df = pd.DataFrame(columns=['image', 'class', 'x', 'y'])

    # 이미지 및 레이블 생성
    for idx, (img, _) in enumerate(src_dataset):
        if idx < index_range[0]:
            continue
        if idx >= index_range[1]:
            break
        if idx % 100 == 0:
            print('generating index:', idx)
        img = img.permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8).copy()
        image_filename = os.path.join(image_path, f'{idx:06d}.png')
        label_filename = os.path.join(label_path, f'{idx:06d}.csv')
        points_data = []

        for color_name, color_value in colors.items():
            num_points = random.randint(0, 5)
            points = draw_random_points(img, num_points, color_value)
            for point in points:
                points_data.append({'image': idx, 'class': color_name, 'x': point[0], 'y': point[1]})

        df_points = pd.DataFrame(points_data)
        if df_points.empty:
            print('EMPTY csv file!', label_filename)
            df_points.to_csv(label_filename, index=False)
        else:
            df_points.to_csv(label_filename, index=False, columns=['class', 'x', 'y'])
        cv2.imwrite(image_filename, img)
        df = pd.concat([df, df_points], ignore_index=True)

    # 전체 데이터프레임 저장
    df.to_csv(os.path.join(data_path, 'labels.csv'), index=False)


def draw_random_points(image, num_points, color):
    h, w, _ = image.shape
    points = []
    for _ in range(num_points):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        points.append((x, y))
        cv2.circle(image, (x, y), radius=3, color=color, thickness=-1)
    return points


if __name__ == '__main__':
    generate_dataset_main()
