from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch

import settings
from examples.example_dataset import DatasetVisualizer
from config.setup_cfg import setup_cfg
from data.loader_factory import loader_factory


class DataLoaderVisualizer(DatasetVisualizer):
    def __init__(self, cfg, split: str):
        super().__init__(cfg, split)
        self.dataloader = loader_factory(cfg, split)
    
    def display_dataset_frames(self):
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f"===== Batch {batch_idx + 1}/{len(self.dataloader)} =====")
            self._print_batch_info(batch_data)
            if self._show_batch(batch_data):
                break
        cv2.destroyAllWindows()
    
    def _print_batch_info(self, batch_data):
        instances = batch_data['instances']
        print(f"\nFirst image in batch:")
        print(f"Filename: {batch_data['filename'][0]}")
        print(f"Image size: {batch_data['width'][0]}x{batch_data['height'][0]}")
        print(f"Number of boxes: {len(instances[0].gt_boxes)}")
        print(f"Class IDs: {instances[0].gt_classes.tolist()}")
    
    def _show_batch(self, batch_data):
        image = self._to_numpy_image(batch_data['image'][0])
        image = self._draw_instances(image, batch_data['instances'][0])        
        cv2.imshow('image', image)
        return cv2.waitKey(0) & 0xFF == ord('q')


def visualize_detection_batch():
    cfg = setup_cfg()
    visualizer = DataLoaderVisualizer(cfg, 'test')
    visualizer.display_dataset_frames()


if __name__ == "__main__":
    visualize_detection_batch()
