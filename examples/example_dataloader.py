from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch

import settings
from examples.example_dataset import DatasetVisualizer
from config.setup_cfg import setup_cfg
from data.loader_factory import build_data_loader


class DataLoaderVisualizer(DatasetVisualizer):
    def __init__(self, cfg, split: str):
        super().__init__(cfg, split)
        self.dataloader = build_data_loader(cfg, split)
    
    def display_dataset_frames(self):
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f"\n===== Batch {batch_idx}/{len(self.dataloader)} =====")
            for i, data in enumerate(batch_data):
                print(f"----- Image {i} in batch:")
                self._print_batch_info(data)
                if self._show_batch(data):
                    return
        cv2.destroyAllWindows()
    
    def _print_batch_info(self, data):
        instances = data['instances']
        print(f"Filename: {data['filename']}")
        print(f"Image size: {data['width']}x{data['height']}")
        print(f"Number of boxes: {len(instances.gt_boxes)}")
        print(f"Class IDs: {instances.gt_classes.tolist()}")
    
    def _show_batch(self, data):
        image = self._to_numpy_image(data['image'])
        image = self._draw_instances(image, data['instances'])        
        cv2.imshow('image', image)
        return cv2.waitKey(0) & 0xFF == ord('q')


def visualize_detection_batch():
    cfg = setup_cfg()
    visualizer = DataLoaderVisualizer(cfg, 'test')
    visualizer.display_dataset_frames()


if __name__ == "__main__":
    visualize_detection_batch()
