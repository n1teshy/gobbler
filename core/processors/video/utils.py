import os
import cv2
from skimage.metrics import structural_similarity as ssim
from typing import Union
import numpy as np


def get_frame_info(path: str) -> tuple[int, int]:
    assert os.path.exists(path), "non-existent video"
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, total_frames
    finally:
        cap.release()


def get_hist_score(img1: Union[str, np.ndarray], img2: Union[str, np.ndarray]) -> float:
    if isinstance(img1, str):
        assert os.path.exists(img1), "non-existent file"
        img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    if isinstance(img2, str):
        assert os.path.exists(img2), "non-existent file"
        img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def get_ssim_score(img1: Union[str, np.ndarray], img2: Union[str, np.ndarray]) -> float:
    if isinstance(img1, str):
        assert os.path.exists(img1), "non-existent file"
        img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    if isinstance(img2, str):
        assert os.path.exists(img2), "non-existent file"
        img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    score, _ = ssim(img1, img2, full=True)
    return score
