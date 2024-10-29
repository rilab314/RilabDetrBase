import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from detectron2.structures import Instances, Boxes

from data.composer_factory import composer_factory


class CustomDetectionDataset(Dataset):
    def __init__(self, cfg, split: str):
        """
        Args:
            cfg.ROOT_PATH: 데이터셋의 루트 경로
            cfg.SPLIT: 'train', 'val', 'test' 중 하나
            cfg.AUGMENT: 데이터 증강 적용 여부 (True/False)
            cfg.INPUT.PIXEL_MEAN: 이미지 정규화에 사용되는 평균 값
            cfg.INPUT.PIXEL_STD: 이미지 정규화에 사용되는 표준편차 값
        """
        split = split.upper()
        self.root_path = cfg.DATASET.ROOT_PATH
        self.split = cfg.DATASET[split].SPLIT
        self.image_dir = str(os.path.join(self.root_path, self.split, 'images'))
        self.label_dir = str(os.path.join(self.root_path, self.split, 'labels'))
        self.image_files = sorted(os.listdir(self.image_dir))
        self.image_files = [file for file in self.image_files if file.endswith('.jpg')]
        self.augment = composer_factory(cfg, split)
        self.cfg = cfg  # cfg 객체를 저장하여 다른 메서드에서 사용

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        bboxes, category_ids = self.load_labels(idx)
        # 데이터 증강 적용
        image, bboxes, category_ids = self.apply_augmentation(
            image, bboxes, category_ids
        )
        # 바운딩 박스와 레이블을 Detectron2의 Instances 객체로 변환
        image_shape = self.cfg.DATASET.IMAGE_HEIGHT, self.cfg.DATASET.IMAGE_WIDTH
        instances = self.create_instances(bboxes, category_ids, image_shape)
        # 최종 데이터 반환
        return {
            'image': image,
            'instances': instances,
            'height': image.shape[1],
            'width': image.shape[2],
            'filename': os.path.join(self.image_dir, self.image_files[idx]),
        }

    def load_image(self, idx):
        """
        Returns:
            image (numpy.ndarray): 로드된 이미지 (RGB)
        """
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_labels(self, idx):
        """
        Returns:
            bboxes (list): 바운딩 박스 리스트 ([[center_x, center_y, width, height], ...])
            category_ids (list): 클래스 레이블 리스트
        """
        image_filename = self.image_files[idx]
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_filename)

        bboxes = []
        category_ids = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id, center_x, center_y, bbox_width, bbox_height = map(float, line.strip().split())
                    bboxes.append([center_x, center_y, bbox_width, bbox_height])
                    category_ids.append(int(class_id))
        else:
            print(f"Label file not found: {label_path}")
        return bboxes, category_ids

    def apply_augmentation(self, image, bboxes, category_ids):
        transformed = self.augment(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        image = transformed['image']
        bboxes = transformed['bboxes']
        category_ids = transformed['category_ids']
        return image, bboxes, category_ids

    def create_instances(self, bboxes, category_ids, image_shape):
        """
        바운딩 박스와 레이블을 Detectron2의 Instances 객체로 변환합니다.

        Args:
            bboxes (list): 바운딩 박스 리스트
            category_ids (list): 클래스 레이블 리스트
            image_shape (tuple): 변환된 이미지의 크기 (height, width)
        Returns:
            instances (Instances): Detectron2 Instances 객체
        """
        image_height, image_width = image_shape
        target = Instances((image_height, image_width))

        if len(bboxes) > 0:
            # YOLO 형식의 바운딩 박스를 Detectron2 형식으로 변환
            boxes = []
            for bbox in bboxes:
                center_x, center_y, bbox_width, bbox_height = bbox
                x_min = (center_x - bbox_width / 2) * image_width
                y_min = (center_y - bbox_height / 2) * image_height
                x_max = (center_x + bbox_width / 2) * image_width
                y_max = (center_y + bbox_height / 2) * image_height
                boxes.append([x_min, y_min, x_max, y_max])
            boxes = torch.tensor(boxes, dtype=torch.float32)
            target.gt_boxes = Boxes(boxes)
            target.gt_classes = torch.tensor(category_ids, dtype=torch.int64)
        else:
            # 바운딩 박스가 없을 경우 빈 텐서를 설정
            target.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            target.gt_classes = torch.tensor([], dtype=torch.int64)

        return target
