from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch

import settings
from examples.example_dataset import DatasetVisualizer
from config.config import load_config
from utility.print_util import print_data


class DataLoaderVisualizer(DatasetVisualizer):
    def __init__(self, cfg, split: str, batch_size: int = 4):
        super().__init__(cfg, split)
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size,
            shuffle=(split.lower() == 'train'),
            num_workers=2,
            collate_fn=lambda x: x
        )
    
    def display_dataset_frames(self):
        for batch_idx, batch_data in enumerate(self.dataloader):
            print(f"===== Batch {batch_idx + 1}/{len(self.dataloader)} =====")
            self._print_frame_info(batch_data[0])
            if self._show_frame(batch_data[0]):
                break
        cv2.destroyAllWindows()


def visualize_detection_batch():
    cfg = load_config()
    visualizer = DataLoaderVisualizer(cfg, 'train', batch_size=4)
    visualizer.display_dataset_frames()


if __name__ == "__main__":
    visualize_detection_batch()

