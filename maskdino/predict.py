import torch
import cv2

import detectron2.data.transforms as T

from maskdino.utils.setup_cfg import setup_cfg
from maskdino.maskdino_model import MaskDINO
from maskdino.utils.print_util import print_structure
from detectron2.engine.defaults import DefaultPredictor
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.structures import Boxes, ImageList, Instances, BitMasks


def predict_main():
    cfg = setup_cfg()
    model = MaskDINO(cfg)
    # print('\n========== model ==========\n', model)
    image = cv2.imread('/home/dolphin/choi_ws/SatLaneDet_2024/maskdino/images/animals.png')
    pred = predict_impl(cfg, model, image)
    print_structure(pred, 'pred')


def predict_impl(cfg, model, original_image):
    model.eval()
    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        if cfg.INPUT.FORMAT == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image.to(cfg.MODEL.DEVICE)
        inputs = {"image": image, "height": height, "width": width}

        predictions = model([inputs])
    return predictions


if __name__ == "__main__":
    predict_main()
