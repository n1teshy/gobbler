import os
from enum import Enum

import appdirs
import pynvml

import gobbler.meta as meta
from gobbler.logger import logger
from gobbler.utils import download


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


def get_yolo_path() -> str:
    path = os.path.join(appdirs.user_data_dir(meta.name))
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "doc_yolo.pt")
    if not os.path.exists(path):
        logger.info("downloading YOLO model to %s", path)
        download(
            "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench/resolve/main/doclayout_yolo_docstructbench_imgsz1024.pt?download=true",
            path,
        )
    return path
