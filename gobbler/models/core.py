import logging
from typing import Optional, Union

import numpy as np
from doclayout_yolo import YOLOv10
from keybert import KeyBERT
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import gobbler.constants as c
import gobbler.globals as glb
from gobbler.logger import logger
from gobbler.models.utils import (
    ClipScene,
    YOLOScene,
    get_cuda_memory,
    get_yolo_path,
    idx_to_clip_scene,
    label_to_yolo_scene,
)
from gobbler.utils import make_pil_images

clip_utils = (None, None)
yolo_model = None
keybert_model = None

clip_scene_descriptions = [
    "a photo of a video conference showing people and/or their profiles",
    "a photo of a diagram or flowchart explaining a process",
    "a photo of tabular data",
    "a photo of lines or paragraphs of text about a topic",
]


def get_clip_utils() -> tuple[CLIPModel, CLIPProcessor]:
    global clip_utils

    if None in clip_utils:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained(
            # TODO: figure out a way to use use_fast=True
            "openai/clip-vit-large-patch14",
            use_fast=False,
        )
        if get_cuda_memory() > 0:
            model = model.to("cuda")
        clip_utils = (model, processor)

    return clip_utils


def run_clip(images: list[Image.Image]) -> list[list[tuple[ClipScene, float]]]:
    model, processor = get_clip_utils()
    inputs = processor(
        text=clip_scene_descriptions,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()
    return [
        sorted(
            [(idx_to_clip_scene(idx), prob) for idx, prob in enumerate(probs)],
            key=lambda x: x[1],
            reverse=True,
        )
        for probs in probs
    ]


def calculate_box_area(x1: float, y1: float, x2: float, y2: float) -> float:
    return (x2 - x1) * (y2 - y1)


def calculate_overlap_metrics(
    box1: tuple, box2: tuple, containment_threshold: float = 0.7
) -> tuple[float, bool]:
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    intersection_x1 = max(x1_1, x1_2)
    intersection_y1 = max(y1_1, y1_2)
    intersection_x2 = min(x2_1, x2_2)
    intersection_y2 = min(y2_1, y2_2)

    if (
        intersection_x1 >= intersection_x2
        or intersection_y1 >= intersection_y2
    ):
        return 0.0, False

    intersection_area = calculate_box_area(
        intersection_x1, intersection_y1, intersection_x2, intersection_y2
    )
    box1_area = calculate_box_area(x1_1, y1_1, x2_1, y2_1)
    box2_area = calculate_box_area(x1_2, y1_2, x2_2, y2_2)
    smaller_area = min(box1_area, box2_area)
    overlap_ratio = (
        intersection_area / smaller_area if smaller_area > 0 else 0.0
    )
    is_contained = (
        (intersection_area / smaller_area) > containment_threshold
        if smaller_area > 0
        else False
    )

    return overlap_ratio, is_contained


def calculate_distance(box1: tuple, box2: tuple) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]

    horizontal_gap = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
    vertical_gap = max(0, max(y1_1, y1_2) - min(y2_1, y2_2))

    return (horizontal_gap**2 + vertical_gap**2) ** 0.5


def should_merge(
    box1: tuple,
    box2: tuple,
    overlap_threshold: float = 0.8,
    proximity_factor: float = 2.0,
) -> bool:
    overlap_ratio, is_contained = calculate_overlap_metrics(
        box1, box2, overlap_threshold
    )
    if overlap_ratio > overlap_threshold or is_contained:
        return True

    label1, label2 = box1[4], box2[4]
    mergeable_groups = [
        {YOLOScene.ISOLATE_FORMULA, YOLOScene.FORMULA_CAPTION},
        {YOLOScene.TABLE, YOLOScene.TABLE_CAPTION, YOLOScene.TABLE_FOOTNOTE},
        {YOLOScene.FIGURE, YOLOScene.FIGURE_CAPTION},
        {YOLOScene.PLAIN_TEXT},
    ]

    for group in mergeable_groups:
        if label1 in group and label2 in group:
            distance = calculate_distance(box1, box2)
            avg_height = (abs(box1[3] - box1[1]) + abs(box2[3] - box2[1])) / 2
            return distance < proximity_factor * avg_height

    if label1 == YOLOScene.TITLE and label2 in {
        YOLOScene.FIGURE,
        YOLOScene.PLAIN_TEXT,
        YOLOScene.TABLE,
    }:
        if box1[3] < box2[1]:
            distance = calculate_distance(box1, box2)
            avg_height = (abs(box1[3] - box1[1]) + abs(box2[3] - box2[1])) / 2
            return distance < proximity_factor * avg_height
    elif label2 == YOLOScene.TITLE and label1 in {
        YOLOScene.FIGURE,
        YOLOScene.PLAIN_TEXT,
        YOLOScene.TABLE,
    }:
        if box2[3] < box1[1]:
            distance = calculate_distance(box1, box2)
            avg_height = (abs(box1[3] - box1[1]) + abs(box2[3] - box2[1])) / 2
            return distance < proximity_factor * avg_height

    return False


def merge_boxes(box1: tuple, box2: tuple) -> tuple:
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    label1, label2 = box1[4], box2[4]
    general_labels = {
        YOLOScene.TABLE,
        YOLOScene.FIGURE,
        YOLOScene.ISOLATE_FORMULA,
        YOLOScene.PLAIN_TEXT,
    }

    if label1 in general_labels:
        label = label1
    elif label2 in general_labels:
        label = label2
    else:
        label = (
            label1
            if calculate_box_area(box1[0], box1[1], box1[2], box1[3])
            > calculate_box_area(box2[0], box2[1], box2[2], box2[3])
            else label2
        )

    confidence = max(box1[5], box2[5])
    return (x1, y1, x2, y2, label, confidence)


def group_related_boxes(
    boxes: list[tuple],
    overlap_threshold: float = 0.8,
    proximity_factor: float = 2.0,
) -> list[tuple]:
    if len(boxes) <= 1:
        return boxes

    remaining_boxes = boxes.copy()
    merged_boxes = []
    while remaining_boxes:
        current_box = remaining_boxes.pop(0)
        merged_any = True
        while merged_any:
            merged_any = False
            for i, other_box in enumerate(remaining_boxes):
                if should_merge(
                    current_box, other_box, overlap_threshold, proximity_factor
                ):
                    current_box = merge_boxes(current_box, other_box)
                    remaining_boxes.pop(i)
                    merged_any = True
                    break
        merged_boxes.append(current_box)
    return merged_boxes


def has_missed_content(
    image: Image.Image, boxes: list[tuple], filled_pixels_stddev: int
) -> bool:
    img_array = np.array(image.convert("L"))
    mask = np.ones(img_array.shape, dtype=bool)
    for x1, y1, x2, y2, _, _ in boxes:
        mask[int(y1) : int(y2), int(x1) : int(x2)] = False

    uncovered_pixels = img_array[mask]
    if len(uncovered_pixels) == 0:
        return False

    background_stddev = np.std(uncovered_pixels)
    return background_stddev > filled_pixels_stddev


def run_yolo(
    paths_or_images: list[Union[str, Image.Image]],
    yolo_threshold: Optional[float] = None,
    fallback_clip_threshold: Optional[float] = None,
    filled_pixels_stddev: Optional[int] = None,
) -> list[
    list[
        tuple[float, float, float, float, Optional[YOLOScene], Optional[float]]
    ]
]:
    """
    Returns `list[tuple[x1, y1, x2, y2, YOLOScene | None, confidence]]` per image.
    """
    global yolo_model
    result = []

    if yolo_model is None:
        # prevents YOLO from polluting stdout
        from doclayout_yolo.utils import LOGGER  # noqa

        LOGGER.setLevel(logging.ERROR)

        yolo_model = YOLOv10(get_yolo_path())

    yolo_threshold = yolo_threshold or glb.yolo_prob_threshold
    fallback_clip_threshold = (
        fallback_clip_threshold or glb.yolo_fallback_clip_threshold
    )
    filled_pixels_stddev = (
        filled_pixels_stddev or glb.filled_pixel_region_stddev
    )

    device = "cuda" if get_cuda_memory() > 0 else "cpu"
    paths_or_images = make_pil_images(paths_or_images)
    bbox_batch = yolo_model.predict(
        paths_or_images, imgsz=1024, conf=yolo_threshold, device=device
    )
    for idx, bbox_data in enumerate(bbox_batch):
        if len(bbox_data.boxes) == 0:
            result.append(
                [
                    (
                        0,
                        0,
                        paths_or_images[idx].width,
                        paths_or_images[idx].height,
                        None,
                        None,
                    )
                ]
            )
            continue

        boxes = []
        for box in bbox_data.boxes:
            xyxy = list(map(int, box.xyxy.tolist()[0]))
            yolo_label = yolo_model.names[box.cls.item()]
            confidence = box.conf.item()
            boxes.append(
                tuple(xyxy + [label_to_yolo_scene(yolo_label), confidence])
            )

        grouped_boxes = group_related_boxes(boxes)
        # TODO: check if some *_caption type boxes were not grouped
        if has_missed_content(
            paths_or_images[idx], grouped_boxes, filled_pixels_stddev
        ):
            logger.info(
                f"YOLO missed content, using CLIP to get page-level category"
            )
            clip_label = None
            scene, prob = run_clip([paths_or_images[idx]])[0][0]
            prob = prob if prob >= fallback_clip_threshold else None
            if prob is not None:
                clip_label = {
                    ClipScene.TABULAR: YOLOScene.TABLE,
                    ClipScene.TEXT: YOLOScene.PLAIN_TEXT,
                    ClipScene.DIAGRAM: YOLOScene.FIGURE,
                }.get(scene, None)
            push_this = (
                0,
                0,
                paths_or_images[idx].width,
                paths_or_images[idx].height,
                clip_label,
                prob,
            )
            result.append([push_this])
        else:
            result.append(grouped_boxes)
    return result


def run_keybert(text: str) -> list[tuple[str, float]]:
    global keybert_model

    if keybert_model is None:
        keybert_model = KeyBERT("bert-base-nli-mean-tokens")
    return keybert_model.extract_keywords(text)
