from transformers import CLIPProcessor, CLIPModel
from typing import Union
from PIL import Image

clip_utils = (None, None)
image_categories = [
    "a photo of profiles on a monochromatic background",
    "a photo of Microsoft PowerPoint",
    "a photo of Microsoft Word",
    "a photo of a diagram or flowchart explaining a process",
    "a photo of lines or paragraphs of text about a topic",
]


def get_clip_utils() -> tuple[CLIPModel, CLIPProcessor]:
    global clip_utils

    if None in clip_utils:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_utils = (model, processor)

    return clip_utils


def classify_images(paths_or_images: list[Union[str, Image.Image]]) -> list[int]:
    for idx in range(len(paths_or_images)):
        if type(paths_or_images[idx]) is str:
            paths_or_images[idx] = Image.open(paths_or_images[idx])

    model, processor = get_clip_utils()
    inputs = processor(
        text=image_categories, images=paths_or_images, return_tensors="pt", padding=True
    )
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs_per_image = logits_per_image.softmax(dim=1).tolist()
    return [
        max(range(len(probs)), key=lambda idx: probs[idx]) for probs in probs_per_image
    ]
