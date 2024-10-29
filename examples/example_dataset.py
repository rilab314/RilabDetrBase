import cv2
import torch
import numpy as np

import settings
from data.custom_detection_dataset import CustomDetectionDataset
from config.setup_cfg import setup_cfg


class DatasetVisualizer:
    def __init__(self, cfg, split: str):
        self.dataset = CustomDetectionDataset(cfg, split)
        self.pixel_mean = cfg.MODEL.PIXEL_MEAN
        self.pixel_std = cfg.MODEL.PIXEL_STD
    
    def display_dataset_frames(self):
        total_frames = len(self.dataset)
        for idx in range(total_frames):
            data = self.dataset[idx]
            print(f"===== Frame {idx + 1}/{total_frames} =====")
            self._print_frame_info(data)
            if self._show_frame(data):
                break
        cv2.destroyAllWindows()
    
    def _print_frame_info(self, data):
        instances = data['instances']
        print(f"Filename: {data['filename']}")
        print(f"Image size: {data['width']}x{data['height']}")
        print(f"Number of boxes: {len(instances.gt_boxes)}")
        print(f"Class IDs: {instances.gt_classes.tolist()}")
        print(f"Box coordinates: {instances.gt_boxes.tensor.tolist()}")
    
    def _show_frame(self, data):
        """이미지에 바운딩 박스와 클래스 ID를 표시합니다.
        
        Args:
            data (dict): 이미지와 바운딩 박스 정보를 포함하는 딕셔너리
        """
        image_tensor = data['image']
        instances = data['instances']
        image = self._to_numpy_image(image_tensor)
        image = self._draw_instances(image, instances)
        cv2.imshow('Frame', image)
        return cv2.waitKey(0) & 0xFF == ord('q')

    def _to_numpy_image(self, image_tensor):
        image = image_tensor.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = ((image * self.pixel_std) + self.pixel_mean) * 255.
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def _draw_instances(self, image, instances):
        boxes = instances.gt_boxes.tensor.detach().cpu().numpy()
        classes = instances.gt_classes.detach().cpu().numpy()        
        for box, class_id in zip(boxes, classes):
            x_min, y_min, x_max, y_max = map(int, box)
            image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{class_id}"
            image = cv2.putText(image, text, (x_min, y_min - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return image


def visualize_detection_dataset():
    cfg = setup_cfg()
    visualizer = DatasetVisualizer(cfg, 'test' )
    visualizer.display_dataset_frames()


if __name__ == "__main__":
    visualize_detection_dataset()
