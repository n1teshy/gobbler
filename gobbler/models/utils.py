from enum import Enum

import numpy as np
import pynvml
import torch
from PIL import Image
from torchvision.ops import batched_nms, nms

from gobbler.logger import logger


class ClipScene(str, Enum):
    VIDEO_CONFERENCE = "video_conference"
    DIAGRAM = "diagram"
    TABULAR = "tabular"
    TEXT = "text"


class YOLOScene(str, Enum):
    TITLE = "title"
    PLAIN_TEXT = "plain text"
    ABANDON = "abandon"
    FIGURE = "figure"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    TABLE_CAPTION = "table_caption"
    TABLE_FOOTNOTE = "table_footnote"
    ISOLATE_FORMULA = "isolate_formula"
    FORMULA_CAPTION = "formula_caption"


clip_scenes = [
    ClipScene.VIDEO_CONFERENCE,
    ClipScene.DIAGRAM,
    ClipScene.TABULAR,
    ClipScene.TEXT,
]
clip_scene_2_idx = {scene: idx for idx, scene in enumerate(clip_scenes)}

yolo_scene_2_label = {scene: scene.value for scene in YOLOScene}
yolo_label_2_scene = {v: k for k, v in yolo_scene_2_label.items()}


def idx_to_clip_scene(idx: int) -> ClipScene:
    return clip_scenes[idx]


def clip_scene_to_idx(scene_type: ClipScene) -> int:
    return clip_scene_2_idx[scene_type]


def label_to_yolo_scene(label: str) -> YOLOScene:
    return yolo_label_2_scene[label]


def yolo_scene_to_label(scene: YOLOScene) -> str:
    return yolo_scene_2_label[scene]


def is_cuda_available() -> bool:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
        return True
    except pynvml.NVMLError_LibraryNotFound:
        return False
    except Exception as e:
        logger.error("uncaught error when initializing pynvml: %s", e)
        return False


def get_cuda_memory() -> int:
    if not is_cuda_available():
        return 0

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.total
    except Exception as e:
        logger.error("uncaught error when getting CUDA memory: %s", e)
        return 0


def preprocess_yolo_images(images: list[Image.Image]) -> torch.Tensor:
    images = [img.convert("RGB").resize((1024, 1024)) for img in images]
    images = np.array(images).astype(np.float32) / 255.0
    images = np.transpose(images, (0, 3, 1, 2))  # [B, C, H, W]
    return torch.from_numpy(images).contiguous()


def postprocess_yolo_preds(
    preds: torch.Tensor,
    conf_thresh: float,
    iou_thres: float = 0.50,
    max_det: int = 300,
    class_agnostic: bool = False,
) -> list[torch.Tensor]:
    if preds.ndim != 3 or preds.shape[-1] < 6:
        raise ValueError(
            f"Expected preds of shape [B, N, 6], got {tuple(preds.shape)}"
        )

    outputs: list[torch.Tensor] = []
    for b in range(preds.shape[0]):
        p = preds[b]
        boxes = p[:, 0:4]
        scores = p[:, 4]
        labels = p[:, 5].to(torch.int64)
        keep = scores >= conf_thresh
        if keep.any():
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        else:
            outputs.append(
                torch.empty((0, 6), dtype=torch.float32, device=preds.device)
            )
            continue
        if class_agnostic:
            keep_idx = nms(boxes, scores, iou_thres)
        else:
            keep_idx = batched_nms(boxes, scores, labels, iou_thres)
        if max_det is not None and keep_idx.numel() > max_det:
            top = torch.topk(scores[keep_idx], k=max_det).indices
            keep_idx = keep_idx[top]
        kept_boxes = boxes[keep_idx]
        kept_scores = scores[keep_idx]
        kept_labels = labels[keep_idx].to(torch.float32).unsqueeze(1)
        out = torch.cat(
            [kept_boxes, kept_labels, kept_scores.unsqueeze(1)], dim=1
        )  # [x1,y1,x2,y2,cls,conf]
        outputs.append(out)

    return outputs
