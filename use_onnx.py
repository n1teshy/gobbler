from enum import Enum

import numpy as np
import onnxruntime as ort
from PIL import Image


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


labels = {
    0: YOLOScene.TITLE,
    1: YOLOScene.PLAIN_TEXT,
    2: YOLOScene.ABANDON,
    3: YOLOScene.FIGURE,
    4: YOLOScene.FIGURE_CAPTION,
    5: YOLOScene.TABLE,
    6: YOLOScene.TABLE_CAPTION,
    7: YOLOScene.TABLE_FOOTNOTE,
    8: YOLOScene.ISOLATE_FORMULA,
    9: YOLOScene.FORMULA_CAPTION,
}

session = ort.InferenceSession(
    "model.onnx", providers=["CPUExecutionProvider"]
)
model_h, model_w = 1024, 1024


def preprocess(imgs):
    if isinstance(imgs, (str, Image.Image)):
        imgs = [imgs]
    batch = []
    for img in imgs:
        if isinstance(img, str):
            img = np.array(Image.open(img).convert("RGB"))
        else:
            img = np.array(img)
        img_resized = np.array(Image.fromarray(img).resize((model_w, model_h)))
        img_input = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        batch.append(img_input)
    return {session.get_inputs()[0].name: np.stack(batch, axis=0)}


def nms(boxes, scores, iou_threshold=0.5):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


def calculate_box_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)


def calculate_distance(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    hg = max(0, max(x1_1, x1_2) - min(x2_1, x2_2))
    vg = max(0, max(y1_1, y1_2) - min(y2_1, y2_2))
    return (hg**2 + vg**2) ** 0.5


def calculate_overlap_metrics(box1, box2, containment_threshold=0.7):
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
    ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0, False
    inter = (ix2 - ix1) * (iy2 - iy1)
    smaller_area = min(
        calculate_box_area(*box1[:4]), calculate_box_area(*box2[:4])
    )
    overlap_ratio = inter / smaller_area if smaller_area > 0 else 0.0
    is_contained = (
        overlap_ratio > containment_threshold if smaller_area > 0 else False
    )
    return overlap_ratio, is_contained


def should_merge(box1, box2, overlap_threshold=0.8, proximity_factor=2.0):
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
            avg_h = (abs(box1[3] - box1[1]) + abs(box2[3] - box2[1])) / 2
            return distance < proximity_factor * avg_h
    if label1 == YOLOScene.TITLE and label2 in {
        YOLOScene.FIGURE,
        YOLOScene.PLAIN_TEXT,
        YOLOScene.TABLE,
    }:
        if box1[1] < box2[1]:
            distance = calculate_distance(box1, box2)
            avg_h = (abs(box1[3] - box1[1]) + abs(box2[3] - box2[1])) / 2
            return distance < proximity_factor * avg_h
    elif label2 == YOLOScene.TITLE and label1 in {
        YOLOScene.FIGURE,
        YOLOScene.PLAIN_TEXT,
        YOLOScene.TABLE,
    }:
        if box2[1] < box1[1]:
            distance = calculate_distance(box1, box2)
            avg_h = (abs(box1[3] - box1[1]) + abs(box2[3] - box2[1])) / 2
            return distance < proximity_factor * avg_h
    return False


def merge_boxes(box1, box2):
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
    if label1 in general_labels and label2 not in general_labels:
        label = label1
    elif label2 in general_labels and label1 not in general_labels:
        label = label2
    else:
        label = (
            label1
            if calculate_box_area(*box1[:4]) > calculate_box_area(*box2[:4])
            else label2
        )
    score = max(box1[5], box2[5])
    return (x1, y1, x2, y2, label, score)


def group_related_boxes(boxes):
    if len(boxes) <= 1:
        return boxes
    remaining = boxes.copy()
    merged = []
    while remaining:
        current = remaining.pop(0)
        merged_any = True
        while merged_any:
            merged_any = False
            for i, other in enumerate(remaining):
                if should_merge(current, other):
                    current = merge_boxes(current, other)
                    remaining.pop(i)
                    merged_any = True
                    break
        merged.append(current)
    return merged


def mask_boxes(image: Image.Image, boxes: list[tuple]):
    arr = np.array(image)
    mask = np.zeros(arr.shape[:2], dtype=bool)
    for x1, y1, x2, y2, *_ in boxes:
        mask[int(y1) : int(y2), int(x1) : int(x2)] = True
    masked = arr.copy()
    masked[mask] = 255
    return Image.fromarray(masked)


def extract_boxes(preds, conf_thresh=0.1, iou_thresh=0.5):
    boxes, scores, classes = preds[:, :4], preds[:, 4], preds[:, 5].astype(int)
    keep = nms(boxes, scores, iou_thresh)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    output = []
    for box, score, cls in zip(boxes, scores, classes):
        if score < conf_thresh:
            continue
        output.append(
            (
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                labels[cls],
                float(score),
            )
        )
    return [b for b in output if b[4] != YOLOScene.ABANDON]


def run_yolo(imgs, conf_thresh=0.1, iou_thresh=0.5, low_effort=False):
    if isinstance(imgs, (str, Image.Image)):
        imgs = [imgs]
    first_preds = session.run(None, preprocess(imgs))[0]
    first_boxes_per_img = [
        extract_boxes(preds, conf_thresh, iou_thresh) for preds in first_preds
    ]
    if low_effort:
        final_boxes_per_img = first_boxes_per_img
    else:
        masked_imgs = [
            mask_boxes(img, boxes) if boxes else img
            for img, boxes in zip(imgs, first_boxes_per_img)
        ]
        second_preds = session.run(None, preprocess(masked_imgs))[0]
        second_boxes_per_img = [
            extract_boxes(preds, conf_thresh, iou_thresh)
            for preds in second_preds
        ]
        final_boxes_per_img = [
            first + second
            for first, second in zip(first_boxes_per_img, second_boxes_per_img)
        ]

    grouped_per_img = [
        group_related_boxes(boxes) for boxes in final_boxes_per_img
    ]
    scaled_results = []
    for img, boxes in zip(imgs, grouped_per_img):
        x_factor, y_factor = img.width / model_w, img.height / model_h
        scaled_boxes = [
            (
                int(x1 * x_factor),
                int(y1 * y_factor),
                int(x2 * x_factor),
                int(y2 * y_factor),
                label,
                score,
            )
            for x1, y1, x2, y2, label, score in boxes
        ]
        scaled_results.append(scaled_boxes)
    return scaled_results
