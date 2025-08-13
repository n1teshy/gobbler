import json
import os
import shutil
from pathlib import Path
from typing import Optional, Union

from openai import AzureOpenAI

import gobbler.constants as c
import gobbler.cred as cred
import gobbler.globals as glb
from gobbler.logger import logger
from gobbler.models.utils import ClipScene
from gobbler.processors.image.core import ImageProcessor
from gobbler.processors.image.models import Image
from gobbler.processors.interfaces import BaseProcessor
from gobbler.processors.video.models import Span
from gobbler.processors.video.utils import (
    extract_frames,
    get_audio,
    get_duration,
    has_audio,
    topic_sys_msg,
)
from gobbler.utils import (
    dump_usage_data,
    get_file_metadata,
    get_usage_file,
)


class VideoProcessor(BaseProcessor):
    def __init__(
        self,
        spf: int = glb.video_seconds_per_frame,
        ssim_threshold: Optional[float] = None,
        hist_threshold: Optional[float] = None,
        scene_to_desc: Optional[dict[ClipScene, str]] = None,
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
        self.ssim_thresh = ssim_threshold or glb.ssim_threshold
        self.hist_thresh = hist_threshold or glb.color_hist_threshold
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
        self.compl_usage_data = {
            cred.AZURE_LLM_MODEL: {
                c.FLD_USAGE_PROMPT: 0,
                c.FLD_USAGE_COMPLETION: 0,
            }
        }
        self.txpn_usage_file = get_usage_file(c.USAGE_AOAI_TRANSCRIPTION)
        self.txpn_usage_data = {cred.AZURE_WHISPER_MODEL: {"seconds": 0}}

    def transcribe(self, path: str) -> list[dict[str, Union[int, str]]]:
        segments = []
        audio_files = get_audio(path)

        try:
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
            # cheeky display of free will
            [os.remove(f) for f, _ in audio_files]

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

    def accumulate_topic_txpn(
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

    def get_spans(self, path: str, metadata: dict) -> list[Span]:
        txpn_segments = self.transcribe(path)
        dur = get_duration(path)
        self.txpn_usage_data[cred.AZURE_WHISPER_MODEL]["seconds"] += dur
        dump_usage_data(self.txpn_usage_data, self.txpn_usage_file)
        topic_time_ranges = self.get_topics(txpn_segments)
        if topic_time_ranges is None:
            raise RuntimeError("Failed to get topics")

        span_dicts = self.accumulate_topic_txpn(
            topic_time_ranges, txpn_segments
        )
        spans: list[Span] = []

        for span_kwargs in span_dicts:
            spans.append(
                Span(
                    **metadata,
                    **span_kwargs,
                )
            )
        return [span for span in spans if span.end - span.start >= self.spf]

    def process_frames(
        self, path: str
    ) -> tuple[list[Image], list[dict[str, float]]]:
        frames, frame_time_ranges = extract_frames(
            path,
            spf=self.spf,
            hist_thresh=self.hist_thresh,
            ssim_thresh=self.ssim_thresh,
        )
        try:
            image_objects, processed_time_ranges = [], []

            for idx in range(len(frames)):
                scene = self.image_processor.classify(frames[idx])
                if scene is ClipScene.VIDEO_CONFERENCE:
                    continue
                image_objects.append(self.image_processor.process(frames[idx]))
                processed_time_ranges.append(frame_time_ranges[idx])
            return image_objects, processed_time_ranges
        finally:
            if frames:
                shutil.rmtree(os.path.dirname(frames[0]))

    def assign_frames(
        self,
        spans: list[Span],
        frames: list[Image],
        frame_ranges: list[dict[str, float]],
    ) -> None:
        for span in spans:
            for frame_idx, frame_range in enumerate(frame_ranges):
                overlap_start = max(frame_range["start"], span.start)
                overlap_end = min(frame_range["end"], span.end)
                overlap_duration = overlap_end - overlap_start

                if overlap_duration >= self.spf:
                    timestamp = (
                        frame_range["start"] + frame_range["end"]
                    ) // 2
                    span.frames[timestamp] = frames[frame_idx]

    def process(
        self, path: str, audio_only: bool = False, frames_only: bool = False
    ) -> Union[list[Span], tuple[list[Image], list[dict[str, float]]]]:
        """
        Returns list[Span] for audio + video or audio_only.
        Returns list[Image] and list[{"start": <float>, "end": <float>}...]
            for frames_only.
        """

        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")

        metadata = get_file_metadata(path)
        if not metadata["mime_type"].startswith("video/"):
            raise ValueError(f"Doesn't seem to be a video {path}")

        if not has_audio(path):
            if not frames_only:
                logger.warning(
                    f"{path} does not have audio, processing in frames_only mode"
                )
                frames_only = True
            if audio_only:
                raise ValueError(
                    f"set 'audio_only=True' but video has not audio track :("
                )

        if frames_only:
            return self.process_frames(path)

        spans = self.get_spans(path, metadata)
        if not audio_only:
            frames, frame_time_ranges = self.process_frames(path)
            self.assign_frames(spans, frames, frame_time_ranges)

        return spans
