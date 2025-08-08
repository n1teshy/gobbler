import os
import subprocess
import tempfile
import warnings
from typing import Union

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import gobbler.globals as glb
from gobbler.utils import temp_file

topic_sys_msg = open(
    os.path.join(glb.instructions_dir, "get_topics.txt"), encoding="utf-8"
).read()


def get_frame_info(path: str) -> tuple[int, int]:
    assert os.path.exists(path), "non-existent video"
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, total_frames
    finally:
        cap.release()


def get_duration(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def remove_duplicates_frames(files: list[str], ssim_thresh: float):
    idx = 1
    last_uniq = files[0]

    while idx < len(files):
        while (
            idx < len(files)
            and get_ssim_score(last_uniq, files[idx]) >= ssim_thresh
        ):
            os.remove(files[idx])
            idx += 1
        if idx >= len(files):
            break
        last_uniq = files[idx]
        idx += 1


def extract_frames(
    path: str, spf: float, hist_thresh: float, ssim_thresh: float
) -> tuple[list[str], list[dict[str, float]]]:
    cap = cv2.VideoCapture(path)
    files = []

    try:
        fps, no_frames = get_frame_info(path)
        frames_dir = tempfile.mkdtemp()
        prev_gray = None

        for index in range(0, no_frames, max(1, int(fps * spf))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                warnings.warn("could not get frame-%d" % (index,))
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                score = get_hist_score(prev_gray, gray)
                if score < hist_thresh:
                    seconds = index / fps
                    file = os.path.join(frames_dir, f"{seconds}.png")
                    cv2.imwrite(file, frame)
                    files.append(file)

            prev_gray = gray

        remove_duplicates_frames(files, ssim_thresh=ssim_thresh)
        files = [f for f in files if os.path.exists(f)]
        video_dur, time_ranges = no_frames / fps, []

        for s_f, e_f in zip(files, files[1:] + [None]):
            s_second = int(os.path.basename(s_f).split(".")[0])
            if e_f is not None:
                e_second = int(os.path.basename(e_f).split(".")[0])
            else:
                e_second = video_dur
            time_ranges.append({"start": s_second, "end": e_second})

        return files, time_ranges
    finally:
        cap.release()


def get_hist_score(
    img1: Union[str, np.ndarray], img2: Union[str, np.ndarray]
) -> float:
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


def get_ssim_score(
    img1: Union[str, np.ndarray], img2: Union[str, np.ndarray]
) -> float:
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


def has_audio(path: str) -> bool:
    assert os.path.exists(path), "non-existent video"
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
    )
    return "audio" in result.stdout


def get_audio(path: str) -> list[tuple[str, float]]:
    MAX_FILE_SIZE = 25 * 1024 * 1024
    audio_files = []
    full_audio_f = temp_file(".wav")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                full_audio_f,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        file_size = os.path.getsize(full_audio_f)
        if file_size <= MAX_FILE_SIZE:
            audio_files.append((full_audio_f, 0.0))
            return audio_files

        total_duration = get_duration(full_audio_f)
        chunk_duration = (total_duration * MAX_FILE_SIZE) / file_size
        chunk_duration *= 0.9
        current_start = 0
        chunk_index = 0

        while current_start < total_duration:
            chunk_end = min(current_start + chunk_duration, total_duration)
            chunk_file = temp_file(f"_chunk_{chunk_index}.wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    path,
                    "-ss",
                    str(current_start),
                    "-t",
                    str(chunk_end - current_start),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    chunk_file,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            audio_files.append((chunk_file, current_start))
            current_start = chunk_end
            chunk_index += 1
        return audio_files
    except:
        for audio_file, _ in audio_files:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        raise
