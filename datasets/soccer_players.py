import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from datasets.composer_factory import composer_factory

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SoccerPlayersDataset(Dataset):
    def __init__(self, cfg, split: str):
        """
        Args:
            cfg.ROOT_PATH: 데이터셋의 루트 경로
            cfg.SPLIT: 'train', 'val', 'test' 중 하나
            cfg.AUGMENT: 데이터 증강 적용 여부 (True/False)
            cfg.INPUT.PIXEL_MEAN: 이미지 정규화에 사용되는 평균 값
            cfg.INPUT.PIXEL_STD: 이미지 정규화에 사용되는 표준편차 값
        """
        self.root_path = cfg.dataset.path
        self.split = split
        self.image_dir = str(os.path.join(cfg.dataset.path, self.split, 'images'))
        self.label_dir = str(os.path.join(cfg.dataset.path, self.split, 'labels'))
        image_files = sorted(os.listdir(self.image_dir))
        self.image_files = [file for file in image_files if file.endswith('.jpg')]
        self.augment = composer_factory(cfg, split)
        self.cfg = cfg  # cfg 객체를 저장하여 다른 메서드에서 사용

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        bboxes, category_ids = self.load_labels(idx)
        image, bboxes, category_ids = self.apply_augmentation(
            image, bboxes, category_ids
        )
        image = image.to(device)
        bboxes = torch.tensor(bboxes, dtype=torch.float32, device=device)
        category_ids = torch.tensor(category_ids, dtype=torch.int64, device=device)
        return {
            'image': image,
            'targets': {'boxes': bboxes, 'labels': category_ids},
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
        return np.array(bboxes, dtype=np.float32), np.array(category_ids, dtype=np.int64)

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



from configs.config import CfgNode


def create_coco_annotations(cfg, split='train', output_json='coco_annotations.json'):
    """
    SoccerPlayersDataset을 순회하여,
    COCO style의 annotation.json 파일을 생성합니다.
    단, targets['boxes']는 [cx, cy, w, h] (0~1 normalize) 형식임.

    Args:
        cfg: config 객체 (cfg.dataset.path, cfg.dataset.num_classes 등 참조)
        split (str): 'train', 'val' 등
        output_json (str): 저장할 json 파일 경로
    """
    dataset = SoccerPlayersDataset(cfg, split=split)
    
    images = []
    annotations = []
    ann_id = 0  # annotation 고유 ID를 0부터 순차 부여

    print(f"Creating COCO annotations for {split}")
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]  # {'image', 'targets': {...}, 'height', 'width', 'filename'}
        filename = sample['filename']
        img_height = sample['height']  # 실제 이미지 세로 픽셀
        img_width  = sample['width']   # 실제 이미지 가로 픽셀

        images.append({
            "file_name": os.path.basename(filename),
            "height": img_height,
            "width": img_width,
            "id": i  # image_id
        })
        boxes = sample["targets"]["boxes"]   # shape: (N, 4) -> [cx, cy, w, h], 0~1 normalized
        labels = sample["targets"]["labels"] # shape: (N,)
        if boxes.numel() == 0:
            continue

        # boxes를 COCO 형식 ([x, y, w, h]) (픽셀 단위)로 변환
        # [cx, cy, w_norm, h_norm] → x_min, y_min, w_pix, h_pix
        for j in range(boxes.shape[0]):
            cx = boxes[j, 0].item()  # normalized center x
            cy = boxes[j, 1].item()  # normalized center y
            bw = boxes[j, 2].item()  # normalized width
            bh = boxes[j, 3].item()  # normalized height
            # 픽셀 단위로 변환
            x_min = (cx - bw/2) * img_width
            y_min = (cy - bh/2) * img_height
            w_box = bw * img_width
            h_box = bh * img_height
            area = w_box * h_box
            category_id = labels[j].item()  # 클래스 ID
            ann = {
                "id": ann_id,
                "image_id": i,
                "category_id": category_id,
                "bbox": [float(x_min), float(y_min), float(w_box), float(h_box)],
                "area": float(area),
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1

    # categories 필드 (각 클래스 id, name)
    # cfg.dataset.num_classes만큼 단순 생성 (id=0..N-1)
    categories = []
    num_classes = getattr(cfg.dataset, "num_classes", 1)
    for cat_id in range(num_classes):
        categories.append({
            "id": cat_id,
            "name": f"class_{cat_id}",
            "supercategory": "none"
        })

    # 최종 COCO 딕셔너리
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)
    print(f"[create_coco_annotations] Saved {len(images)} images, {len(annotations)} annotations -> {output_json}")


if __name__ == "__main__":
    cfg = CfgNode.from_file('defm_detr_base')
    out_file = os.path.join(cfg.dataset.path, 'train', 'instances_train.json')
    create_coco_annotations(cfg=cfg, split='train', output_json=out_file)
    out_file = os.path.join(cfg.dataset.path, 'val', 'instances_val.json')
    create_coco_annotations(cfg=cfg, split='val', output_json=out_file)

