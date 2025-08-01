import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Union

import cv2
from openai import AzureOpenAI

import gobbler.constants as c
import gobbler.cred as cred
from gobbler.logger import logger
from gobbler.processors.image.core import ImageProcessor
from gobbler.processors.image.models import Image
from gobbler.processors.image.utils import SceneType
from gobbler.processors.interfaces import BaseProcessor
from gobbler.processors.video.models import Span
from gobbler.processors.video.utils import (
    get_frame_info,
    get_hist_score,
    get_ssim_score,
    topic_sys_msg,
)
from gobbler.utils import (
    dump_usage_data,
    get_file_metadata,
    get_usage_file,
    load_usage_data,
    temp_file,
)


class VideoProcessor(BaseProcessor):
    def __init__(
        self,
        spf: int = c.SECONDS_PER_FRAME,
        ssim_threshold: float = c.SSIM_THRESH,
        hist_threshold: float = c.NON_SCENIC_HIST_THRESH,
        scene_to_desc: Optional[dict[SceneType, str]] = None,
    ):
        """
        Parameters:
        - spf: Seconds per frame, one frame per <spf> seconds will be taken during frame-skipping.
        - ssim_threshold: Threshold for SSIM to consider frames as different. Default is 0.9.
        - hist_threshold: Threshold for histogram comparison to consider frames as different. Default is 0.99.
        - scene_to_desc: User-provided mapping of scene types to descriptions, OCR API will not be used for these scene types.
        """
        if cred.AZURE_WHISPER_KEY is None:
            raise EnvironmentError("Missing Azure Whisper key")
        if cred.AZURE_LLM_KEY is None:
            raise EnvironmentError("Missing Azure LLM key")
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError(
                "ffmpeg is not installed or not found in PATH"
            )

        self.spf = spf
        self.ssim_thresh = ssim_threshold
        self.hist_thresh = hist_threshold
        self.frames_dir: Optional[tempfile.NamedTemporaryFile] = None
        self.image_processor = ImageProcessor(
            scene_to_desc=scene_to_desc or {}
        )
        self.llm_client = AzureOpenAI(
            api_key=cred.AZURE_LLM_KEY,
            azure_deployment=cred.AZURE_LLM_DEPLOYMENT,
            azure_endpoint=cred.AZURE_LLM_BASE,
            api_version=cred.AZURE_LLM_VERSION,
        )
        self.whisper_client = AzureOpenAI(
            api_key=cred.AZURE_WHISPER_KEY,
            azure_endpoint=cred.AZURE_WHISPER_BASE,
            azure_deployment=cred.AZURE_WHISPER_DEPLOYMENT,
            api_version=cred.AZURE_WHISPER_VERSION,
        )
        self.compl_usage_file = get_usage_file(c.USAGE_AOAI_COMPLETION)
        self.compl_usage_data = load_usage_data(self.compl_usage_file)
        self.txpn_usage_file = get_usage_file(c.USAGE_AOAI_TRANSCRIPTION)
        self.txpn_usage_data = load_usage_data(self.txpn_usage_file)
        self.compl_usage_data[cred.AZURE_LLM_MODEL] = (
            self.compl_usage_data.get(
                cred.AZURE_LLM_MODEL,
                {c.FLD_USAGE_PROMPT: 0, c.FLD_USAGE_COMPLETION: 0},
            )
        )
        self.txpn_usage_data[cred.AZURE_WHISPER_MODEL] = (
            self.txpn_usage_data.get(cred.AZURE_WHISPER_MODEL, {"seconds": 0})
        )

    def remove_duplicates_frames(self, files: list[str], show_progress: bool):
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

    def extract_frames(
        self, path: str, show_progress: bool
    ) -> tuple[list[str], list[dict[str, float]], float]:
        cap = cv2.VideoCapture(path)
        files = []

        try:
            fps, no_frames = get_frame_info(path)
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
                        file = os.path.join(
                            self.frames_dir.name, f"{seconds}.png"
                        )
                        cv2.imwrite(file, frame)
                        files.append(file)

                if show_progress:
                    print(
                        f"--- frame extraction: {((index + 1) / no_frames * 100):.2f}% ---",
                        end="\r",
                    )
                prev_gray = gray

            self.remove_duplicates_frames(files, show_progress)
            files = [f for f in files if os.path.exists(f)]
            video_dur, time_ranges = no_frames / fps, []

            for s_f, e_f in zip(files, files[1:] + [None]):
                s_second = int(os.path.basename(s_f).split(".")[0])
                if e_f is not None:
                    e_second = int(os.path.basename(e_f).split(".")[0])
                else:
                    e_second = video_dur
                time_ranges.append({"start": s_second, "end": e_second})

            return files, time_ranges, no_frames / fps
        finally:
            cap.release()

    def transcribe(self, path: str) -> list[dict[str, Union[int, str]]]:
        segments = []
        audio_files = []

        try:
            audio_files = self.get_audio(path)
            for audio_file, start_offset in audio_files:
                transcription = (
                    self.whisper_client.audio.transcriptions.create(
                        model=cred.AZURE_WHISPER_MODEL,
                        file=Path(audio_file),
                        response_format="verbose_json",
                    )
                )

                for seg in transcription.segments:
                    segments.append(
                        {
                            "start": seg.start + start_offset,
                            "end": seg.end + start_offset,
                            "text": seg.text,
                        }
                    )

            return segments
        finally:
            for audio_file, _ in audio_files:
                os.remove(audio_file)

    def get_topics(
        self, segments: list[dict[str, Union[int, str]]]
    ) -> Optional[list[dict[str, Union[int, str]]]]:
        text = "\n".join(
            [
                "%d - %d => %s" % (seg["start"], seg["end"], seg["text"])
                for seg in segments
            ]
        )
        for _ in range(3):
            response = self.llm_client.chat.completions.create(
                model=cred.AZURE_LLM_MODEL,
                messages=[
                    {
                        c.LLM_FLD_ROLE: c.LLM_ROLE_SYSTEM,
                        c.LLM_FLD_CONTENT: topic_sys_msg,
                    },
                    {c.LLM_FLD_ROLE: c.LLM_ROLE_USER, c.LLM_FLD_CONTENT: text},
                ],
                response_format={c.LLM_FLD_TYPE: "json_object"},
            )
            self.compl_usage_data[cred.AZURE_LLM_MODEL][
                c.FLD_USAGE_PROMPT
            ] += response.usage.prompt_tokens
            self.compl_usage_data[cred.AZURE_LLM_MODEL][
                c.FLD_USAGE_COMPLETION
            ] += response.usage.completion_tokens
            dump_usage_data(self.compl_usage_data, self.compl_usage_file)
            data = json.loads(response.choices[0].message.content)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "key_segments" in data:
                return data["key_segments"]
            else:
                logger.warning(f"Unexpected response format: {data}")
                continue

    def get_spans(
        self,
        time_ranges: list[dict[str, float]],
        segments: list[dict[str, Union[float, str]]],
    ) -> list[dict]:
        span_data = []
        for range in time_ranges:
            start, end, short_desc, keywords = (
                range["start"],
                range["end"],
                range["short_description"],
                range["keywords"],
            )
            start_idx, end_idx, seg_idx = 0, len(segments) - 1, 0
            while seg_idx < len(segments):
                if abs(segments[seg_idx]["start"] - start) <= 1:
                    start_idx = seg_idx
                    break
                if segments[seg_idx]["start"] > start:
                    start_idx = seg_idx - 1
                    break
                seg_idx += 1
            start_idx = seg_idx
            while seg_idx < len(segments):
                if abs(segments[seg_idx]["end"] - end) <= 1:
                    end_idx = seg_idx
                    break
                if segments[seg_idx]["start"] > end:
                    end_idx = seg_idx - 1
                    break
                seg_idx += 1
            # not joining with ' ' because whisper ensures that
            text = "".join(
                s["text"]
                for s in segments[start_idx : max(start_idx, end_idx) + 1]
            ).strip()
            span_kwargs = dict(
                start=start,
                end=end,
                short_description=short_desc,
                long_description=text,
                keywords=keywords,
            )
            span_data.append(span_kwargs)
        return span_data

    def assign_frames(
        self,
        spans: list[Span],
        frames: list[str],
        frame_ranges: list[dict],
        show_progress: bool = False,
    ) -> None:
        frame_idx = 0
        processed_images = {}

        for span in spans:
            while (
                0 <= frame_idx < len(frame_ranges)
                and frame_ranges[frame_idx]["start"] > span.start
            ):
                frame_idx -= 1
            if (
                0 <= frame_idx < len(frame_ranges)
                and frame_ranges[frame_idx]["start"] < span.start
            ):
                frame_idx += 1
            while (
                0 <= frame_idx < len(frame_ranges)
                and min(frame_ranges[frame_idx]["end"], span.end)
                - max(frame_ranges[frame_idx]["start"], span.start)
                >= self.spf
            ):
                scene = self.image_processor.classify(frames[frame_idx])
                if scene is not SceneType.VIDEO_CONFERENCE:
                    if frame_idx not in processed_images:
                        if show_progress:
                            logger.info(
                                f"--- describing {frames[frame_idx]} ---"
                            )
                        processed_images[frame_idx] = (
                            self.image_processor.process(
                                frames[frame_idx], scene
                            )
                        )
                    span.frames.append(processed_images[frame_idx])
                frame_idx += 1

    def process(
        self, path: str, audio_only: bool = False, show_progress: bool = False
    ) -> list[Span]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        metadata = get_file_metadata(path)
        if not metadata["mime_type"].startswith("video/"):
            raise ValueError(f"Doesn't seem to be a video {path}")

        if show_progress:
            logger.info(f"--- transcribing ---")
        txpn_segments = self.transcribe(path)
        self.txpn_usage_data[cred.AZURE_WHISPER_MODEL]["seconds"] += dur
        dump_usage_data(self.txpn_usage_data, self.txpn_usage_file)
        topic_time_ranges = self.get_topics(txpn_segments)

        if topic_time_ranges is None:
            raise RuntimeError("Failed to get topics")

        span_dicts = self.get_spans(topic_time_ranges, txpn_segments)
        spans: list[Span] = []

        for span_kwargs in span_dicts:
            spans.append(
                Span(
                    **metadata,
                    **span_kwargs,
                )
            )

        spans = [span for span in spans if span.end - span.start >= self.spf]
        if audio_only:
            return spans

        frames, frame_time_ranges, dur = self.extract_frames(
            path, show_progress
        )
        self.assign_frames(spans, frames, frame_time_ranges, show_progress)
        return spans

    def cleanup(self):
        if self.frames_dir is not None:
            self.frames_dir.cleanup()

    def get_audio(self, path: str) -> list[tuple[str, float]]:
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

            duration_result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    full_audio_f,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            total_duration = float(duration_result.stdout.strip())
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
