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


idx_2_scenes = [
    ClipScene.VIDEO_CONFERENCE,
    ClipScene.DIAGRAM,
    ClipScene.TABULAR,
    ClipScene.TEXT,
]
scenes_2_idx = {scene: idx for idx, scene in enumerate(idx_2_scenes)}


def idx_to_clip_scene(idx: int) -> ClipScene:
    return idx_2_scenes[idx]


def clip_scene_to_idx(scene_type: ClipScene) -> int:
    return scenes_2_idx[scene_type]


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
