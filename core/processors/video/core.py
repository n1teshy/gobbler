import os
import cv2
import tempfile
from typing import Optional
import warnings

from core.processors.video.utils import get_frame_info, get_hist_score, get_ssim_score
from core.processors.interfaces import BaseProcessor
import core.constants as c


class VideoProcessor(BaseProcessor):
    def __init__(
        self,
        path: str,
        spf: int = c.SECONDS_PER_FRAME,
        ssim_threshold: float = c.SSIM_THRESH,
        hist_threshold: float = c.NON_SCENIC_HIST_THRESH,
    ):
        self.path = path
        self.spf = spf
        self.ssim_thresh = ssim_threshold
        self.hist_thresh = hist_threshold
        self.frames_dir: Optional[tempfile.NamedTemporaryFile] = None

    def remove_duplicates(self, files: list[str], show_progress: bool):
        idx = 1
        last_uniq = files[0]

        while idx < len(files):
            while (
                idx < len(files)
                and get_ssim_score(last_uniq, files[idx]) >= self.ssim_thresh
            ):
                os.remove(files[idx])
                idx += 1
            if show_progress:
                print(
                    f"--- frame de-duplication: {min((idx + 1) / len(files) * 100, 100):.2f}% ---",
                    end="\r",
                )
            if idx >= len(files):
                break
            last_uniq = files[idx]
            idx += 1

    def extract_frames(self, show_progress: bool):
        cap = cv2.VideoCapture(self.path)
        files = []

        try:
            fps, no_frames = get_frame_info(self.path)
            self.frames_dir = tempfile.TemporaryDirectory()
            prev_gray = None

            for index in range(0, no_frames, max(1, int(fps * self.spf))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = cap.read()
                if not ret:
                    warnings.warn("could not get frame-%d" % (index,))
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    score = get_hist_score(prev_gray, gray)
                    if score < self.hist_thresh:
                        seconds = index / fps
                        h = int(seconds // 3600)
                        m = int((seconds % 3600) // 60)
                        s = int(seconds % 60)
                        timestamp = f"{h:02}-{m:02}-{s:02}"
                        file = os.path.join(self.frames_dir.name, f"{timestamp}.png")
                        cv2.imwrite(file, frame)
                        files.append(file)

                if show_progress:
                    print(
                        f"--- frame extraction: {((index + 1) / no_frames * 100):.2f}% ---",
                        end="\r",
                    )
                prev_gray = gray
            self.remove_duplicates(files, show_progress)
        finally:
            cap.release()

    def process(self, show_progress: bool = False):
        self.extract_frames(show_progress)

    def cleanup(self):
        if self.frames_dir is not None:
            self.frames_dir.cleanup()
