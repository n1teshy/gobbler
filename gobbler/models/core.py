from typing import TYPE_CHECKING, Union

from doclayout_yolo import YOLOv10
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from gobbler.models.utils import get_cuda_memory, get_yolo_path

if TYPE_CHECKING:
    import torch

clip_utils = (None, None)
yolo_model = None


def get_clip_utils() -> tuple[CLIPModel, CLIPProcessor]:
    global clip_utils

    if None in clip_utils:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        if get_cuda_memory() > 0:
            model = model.to("cuda")
        clip_utils = (model, processor)

    return clip_utils


def run_clip(
    descriptions: list[str], images: list[Image.Image]
) -> "torch.Tensor":
    """
    Returns probabilities
    [[img1_prob1, img1_prob2], [img1_prob1, img1_prob2]]
    """
    model, processor = get_clip_utils()
    inputs = processor(
        text=descriptions,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image.softmax(dim=1).tolist()


def run_yolo(
    images: list[Image.Image], confidence_threshold: float = 0.5
) -> list[list[tuple[float, float, float, float, str, float]]]:
    """
    Returns `list[tuple[x1, y1, x2, y2, label, confidence]]` per image.
    `label` can be one of 'title', 'plain text', 'abandon', 'figure', 'figure_caption',
    'table', 'table_caption', 'table_footnote', 'isolate_formula' and 'formula_caption'.
    """
    global yolo_model
    result = []

    if yolo_model is None:
        yolo_model = YOLOv10(get_yolo_path())
    device = "cuda" if get_cuda_memory() > 0 else "cpu"
    batch = yolo_model.predict(
        images, imgsz=1024, conf=confidence_threshold, device=device
    )
    for img_data in batch:
        result.append([])
        for box in img_data.boxes:
            xyxyn = box.xyxyn.tolist()[0]
            label = yolo_model.names[box.cls.item()]
            confidence = box.conf.item()
            result[-1].append(tuple(xyxyn + [label, confidence]))
    return result
