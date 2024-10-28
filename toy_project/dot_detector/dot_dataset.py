import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class DotDataset(Dataset):
    def __init__(self, dataset_path, image_resolution, max_label):
        self.image_path = os.path.join(dataset_path, 'image')
        self.label_path = os.path.join(dataset_path, 'label')
        self.image_files = sorted(os.listdir(self.image_path))
        self.label_files = sorted(os.listdir(self.label_path))
        self.image_resolution = image_resolution
        self.max_label = max_label
        self.color_map = {'red': 1, 'green': 2, 'blue': 3}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지 로드 및 전처리
        img_file = self.image_files[idx]
        img = cv2.imread(os.path.join(self.image_path, img_file))
        img = cv2.resize(img, (self.image_resolution[1], self.image_resolution[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        img = img / 255.0  # Normalize to [0, 1]
        img = torch.tensor(img, dtype=torch.float32)

        # 레이블 로드 및 전처리
        lbl_file = self.label_files[idx]
        labels = pd.read_csv(os.path.join(self.label_path, lbl_file))

        label_tensor = torch.zeros((3, self.max_label), dtype=torch.float32)
        for i, (_, row) in enumerate(labels.iterrows()):
            if i >= self.max_label:
                break
            class_idx = self.color_map[row['class']]
            x, y = row['x'], row['y']
            label_tensor[0, i] = x
            label_tensor[1, i] = y
            label_tensor[2, i] = class_idx

        return img, label_tensor


def example_dataset():
    # 사용 예제
    image_resolution = (512, 512)  # Dataset에서 출력해야 하는 이미지 해상도
    max_label = 20  # 한 영상에서 최대한 받을 수 있는 라벨 수, 20개 미만이라도 zero-padding으로 20개를 내보내야 함
    dataset_path = '/home/dolphin/choi_ws/SatLaneDet_2024/dataset/train'  # 데이터셋이 저장된 경로

    dataset = DotDataset(dataset_path, image_resolution, max_label)
    for image, label in dataset:
        print(f"image.shape={image.shape}, label.shape={label.shape}")
        print('label\n', label[:, :10])
        break  # 예제에서는 첫 번째 데이터만 확인


if __name__ == '__main__':
    example_dataset()
