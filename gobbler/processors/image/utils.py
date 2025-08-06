import os.path as path
from enum import Enum
from typing import Union

from PIL import Image

import gobbler.globals as glb
from gobbler.models.core import run_clip

clip_utils = (None, None)
image_categories = [
    "a photo of a video conference showing people and/or their profiles",
    "a photo of a diagram or flowchart explaining a process",
    "a photo of tabular data",
    "a photo of lines or paragraphs of text about a topic",
]
sys_msg_dsc_diagram = open(
    path.join(glb.instructions_dir, "describe_diagram.txt"), "r"
).read()
sys_msg_dsc_entities = open(
    path.join(glb.instructions_dir, "describe_entity_s.txt"), "r"
).read()
sys_msg_desc_text = open(
    path.join(glb.instructions_dir, "describe_diagram.txt"), "r"
).read()


class SceneType(str, Enum):
    VIDEO_CONFERENCE = "video_conference"
    DIAGRAM = "diagram"
    TABULAR = "tabular"
    TEXT = "text"


idx_2_scenes = [
    SceneType.VIDEO_CONFERENCE,
    SceneType.DIAGRAM,
    SceneType.TABULAR,
    SceneType.TEXT,
]
scenes_2_idx = {scene: idx for idx, scene in enumerate(idx_2_scenes)}


def idx_to_scene_type(idx: int) -> SceneType:
    return idx_2_scenes[idx]


def scene_type_to_idx(scene_type: SceneType) -> int:
    return scenes_2_idx[scene_type]


def classify_images(
    paths_or_images: list[Union[str, Image.Image]],
) -> list[list[tuple[SceneType, float]]]:
    for idx in range(len(paths_or_images)):
        if type(paths_or_images[idx]) is str:
            paths_or_images[idx] = Image.open(paths_or_images[idx])
    probs_per_image = run_clip(image_categories, paths_or_images)
    return [
        sorted(
            [(idx_to_scene_type(idx), prob) for idx, prob in enumerate(probs)],
            key=lambda x: x[1],
            reverse=True,
        )
        for probs in probs_per_image
    ]
