import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import tqdm

import gobbler.meta as meta
from gobbler.logger import logger
from gobbler.processors.docs.core import DocumentProcessor
from gobbler.processors.image.core import ImageProcessor
from gobbler.processors.video.core import VideoProcessor
from gobbler.utils import get_mime_type

BANNER = f"""
   _____       _     _     _
  / ____|     | |   | |   | |
 | |  __  ___ | |__ | |__ | | ___ _ __
 | | |_ |/ _ \| '_ \| '_ \| |/ _ \ '__|
 | |__| | (_) | |_) | |_) | |  __/ |
  \_____|\___/|_.__/|_.__/|_|\___|_|   v{meta.version}
"""

SUPPORTED_EXTENSIONS = {
    "document": {".pdf", ".doc", ".docx", ".ppt", ".pptx"},
    "image": {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"},
}


def clear_screen():
    print("\033[2J\033[H", end="", flush=True)


def load_processing_state(state_file: str) -> dict[str, List[str]]:
    if not os.path.exists(state_file):
        return {"processed": [], "failed": []}

    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_processing_state(
    state_file: str, processed: List[str], failed: List[Tuple[str, str]]
):
    state = {
        "processed": processed,
        "failed": [{"file": f, "error": e} for f, e in failed],
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def collect_usage_stats(processors: dict) -> dict:
    stats = {}

    for proc_class, processor in processors.items():
        proc_stats = {}

        if proc_class == "ImageProcessor":
            if hasattr(processor, "usage_data") and processor.usage_data:
                proc_stats["vision"] = processor.usage_data

        elif proc_class == "VideoProcessor":
            if (
                hasattr(processor, "compl_usage_data")
                and processor.compl_usage_data
            ):
                proc_stats["completion"] = processor.compl_usage_data
            if (
                hasattr(processor, "txpn_usage_data")
                and processor.txpn_usage_data
            ):
                proc_stats["transcription"] = processor.txpn_usage_data

        if proc_stats:
            stats[proc_class] = proc_stats

    return stats


def display_usage_stats(stats: dict[str, Any]):
    if not stats:
        return

    print("\n" + "=" * 60)
    print("API USAGE STATISTICS")
    print("=" * 60)

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_seconds = 0

    for processor_class, processor_stats in stats.items():
        print(f"\n{processor_class}:")

        for api_type, models_data in processor_stats.items():
            print(f"  {api_type.capitalize()} API:")

            for model, usage in models_data.items():
                print(f"    {model}:")

                if "prompt_tokens" in usage:
                    prompt_tokens = usage["prompt_tokens"]
                    completion_tokens = usage.get("completion_tokens", 0)
                    print(f"      Prompt tokens: {prompt_tokens:,}")
                    print(f"      Completion tokens: {completion_tokens:,}")
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens

                if "seconds" in usage:
                    seconds = usage["seconds"]
                    print(f"      Audio seconds: {seconds:,.1f}")
                    total_seconds += seconds

    print(f"\nTOTAL USAGE:")
    if total_prompt_tokens > 0 or total_completion_tokens > 0:
        print(f"  Prompt tokens: {total_prompt_tokens:,}")
        print(f"  Completion tokens: {total_completion_tokens:,}")
        print(
            f"  Total tokens: {(total_prompt_tokens + total_completion_tokens):,}"
        )
    if total_seconds > 0:
        print(f"  Audio seconds: {total_seconds:,.1f}")
    print("=" * 60)


def get_processor_for_file(file_path: str) -> Optional[type]:
    mime_type = get_mime_type(file_path)
    if not mime_type:
        return None

    extension = Path(file_path).suffix.lower()

    if (
        mime_type.startswith("image/")
        or extension in SUPPORTED_EXTENSIONS["image"]
    ):
        return ImageProcessor
    elif (
        mime_type.startswith("video/")
        or extension in SUPPORTED_EXTENSIONS["video"]
    ):
        return VideoProcessor
    elif (
        mime_type == "application/pdf"
        or extension in SUPPORTED_EXTENSIONS["document"]
    ):
        return DocumentProcessor

    return None


def collect_files(input_path: str, recursive: bool = True) -> List[str]:
    files = []
    input_path_obj = Path(input_path)

    if input_path_obj.is_file():
        if get_processor_for_file(input_path):
            files.append(input_path)
    elif input_path_obj.is_dir():
        pattern = "**/*" if recursive else "*"
        for file_path in input_path_obj.glob(pattern):
            if file_path.is_file() and get_processor_for_file(str(file_path)):
                files.append(str(file_path))

    return files


def ensure_output_structure(
    input_path: str, output_dir: str, file_path: str
) -> str:
    input_path_obj = Path(input_path)
    file_path_obj = Path(file_path)
    output_dir_obj = Path(output_dir)

    if input_path_obj.is_file():
        return str(output_dir_obj)

    relative_path = file_path_obj.relative_to(input_path_obj)
    output_subdir = output_dir_obj / relative_path.parent
    output_subdir.mkdir(parents=True, exist_ok=True)

    return str(output_subdir)


def process_single_file(
    file_path: str,
    output_dir: str,
    processors: dict[str, Any],
    progress_bar: Optional[tqdm.tqdm] = None,
) -> Tuple[bool, Optional[str]]:
    try:
        processor_class = get_processor_for_file(file_path)
        if not processor_class:
            error_msg = f"No processor found for file: {file_path}"
            logger.warning(error_msg)
            return False, error_msg

        processor = processors[processor_class.__name__]

        if progress_bar:
            file_name = Path(file_path).name
            progress_bar.set_description(f"{file_name}")

        if processor_class == VideoProcessor:
            results = processor.process(file_path)
        elif processor_class == ImageProcessor:
            results = [processor.process(file_path)]
        elif processor_class == DocumentProcessor:
            results = processor.process(file_path)

        file_stem = Path(file_path).stem
        output_file = Path(output_dir) / f"{file_stem}.json"

        json_results = [result.to_json() for result in results]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                json_results, f, indent=2, ensure_ascii=False, default=str
            )

        logger.info(f"Processed {file_path} -> {output_file}")
        return True, None

    except Exception as e:
        error_msg = f"Failed to process {file_path}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def main():
    clear_screen()
    print(BANNER)

    parser = argparse.ArgumentParser(
        prog=meta.name,
        description="Process documents, images, and videos with AI-powered analysis",
        formatter_class=lambda prog: argparse.HelpFormatter(
            prog, max_help_position=25, width=120
        ),
        usage=argparse.SUPPRESS,
    )

    parser.add_argument("input", help="Input file or directory to process")

    parser.add_argument(
        "-o",
        "--output",
        default="./output",
        help="Output directory for JSON results",
    )

    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't process subdirectories recursively",
    )

    img_group = parser.add_argument_group("Image Processing Options")
    img_group.add_argument(
        "--clip-prob-threshold",
        type=float,
        help="CLIP probability threshold for scene classification",
    )
    img_group.add_argument(
        "--image-heur-threshold",
        type=float,
        help="Heuristic threshold for image classification",
    )

    vid_group = parser.add_argument_group("Video Processing Options")
    vid_group.add_argument(
        "--seconds-per-frame", type=int, help="Extract one frame per N seconds"
    )
    vid_group.add_argument(
        "--ssim-threshold",
        type=float,
        help="SSIM threshold for frame similarity",
    )
    vid_group.add_argument(
        "--color-hist-threshold",
        type=float,
        help="Color histogram threshold for frame similarity",
    )
    vid_group.add_argument(
        "--audio-only",
        action="store_true",
        help="Process only audio from video files",
    )
    vid_group.add_argument(
        "--frames-only",
        action="store_true",
        help="Process only frames from video files",
    )

    doc_group = parser.add_argument_group("Document Processing Options")
    doc_group.add_argument(
        "--yolo-prob-threshold",
        type=float,
        help="YOLO probability threshold for object detection",
    )
    doc_group.add_argument(
        "--yolo-fallback-clip-threshold",
        type=float,
        help="YOLO fallback CLIP threshold",
    )
    doc_group.add_argument(
        "--filled-pixel-region-stddev",
        type=int,
        help="Standard deviation for filled pixel regions",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress bars and verbose output",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what files would be processed without actually processing them",
    )

    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Don't save/load processing state (process all files even if previously processed)",
    )

    parser.add_argument(
        "--state-file",
        default=".gobbler_state.json",
        help="File to save processing state",
    )

    args = parser.parse_args(sys.argv[1:] or ["-h"])

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)

    import gobbler.globals as glb

    if args.clip_prob_threshold is not None:
        glb.clip_prob_thresh = args.clip_prob_threshold
    if args.ssim_threshold is not None:
        glb.ssim_threshold = args.ssim_threshold
    if args.color_hist_threshold is not None:
        glb.color_hist_threshold = args.color_hist_threshold
    if args.seconds_per_frame is not None:
        glb.video_seconds_per_frame = args.seconds_per_frame
    if args.yolo_prob_threshold is not None:
        glb.yolo_prob_threshold = args.yolo_prob_threshold
    if args.yolo_fallback_clip_threshold is not None:
        glb.yolo_fallback_clip_threshold = args.yolo_fallback_clip_threshold
    if args.filled_pixel_region_stddev is not None:
        glb.filled_pixel_region_stddev = args.filled_pixel_region_stddev

    files = collect_files(args.input, recursive=not args.no_recursive)

    if not files:
        print(f"No processable files found in '{args.input}'")
        print(
            f"Supported extensions: {', '.join(sum(SUPPORTED_EXTENSIONS.values(), set()))}"
        )
        sys.exit(0)

    processed_files = []
    failed_files = []

    if not args.no_state:
        state = load_processing_state(args.state_file)
        processed_files = state.get("processed", [])
        failed_files = [
            (f["file"], f["error"]) for f in state.get("failed", [])
        ]

        files_to_process = [f for f in files if f not in processed_files]
        skipped_count = len(files) - len(files_to_process)

        if skipped_count > 0:
            print(f"Skipping {skipped_count} already processed files")

        files = files_to_process

        if failed_files:
            print(f"\nFound {len(failed_files)} previously failed files:")
            for file_path, error in failed_files:
                print(f"  {file_path}: {error}")

            retry = input("\nRetry failed files? (y/N): ").lower().strip()
            if retry == "y":
                files.extend([f for f, _ in failed_files])
                failed_files = []
            else:
                print("Skipping previously failed files")

    if not files:
        print("No files to process")
        sys.exit(0)

    if args.dry_run:
        print(f"\nWould process {len(files)} files:")
        for file_path in files:
            processor_class = get_processor_for_file(file_path)
            processor_name = (
                processor_class.__name__ if processor_class else "unknown"
            )
            print(f"  {file_path} -> {processor_name}")
        sys.exit(0)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    processors = {}

    try:
        print("Initializing processors...")

        if any(get_processor_for_file(f) == ImageProcessor for f in files):
            processors["ImageProcessor"] = ImageProcessor()

        if any(get_processor_for_file(f) == VideoProcessor for f in files):
            import gobbler.globals as glb

            processors["VideoProcessor"] = VideoProcessor(
                spf=glb.video_seconds_per_frame,
                ssim_threshold=glb.ssim_threshold,
                hist_threshold=glb.color_hist_threshold,
            )

        if any(get_processor_for_file(f) == DocumentProcessor for f in files):
            processors["DocumentProcessor"] = DocumentProcessor()

    except Exception as e:
        logger.error(f"Failed to initialize processors: {e}")
        sys.exit(1)

    print(f"Processing {len(files)} files...")

    processed_count = 0
    current_failed = []

    if args.quiet:
        for file_path in files:
            output_subdir = ensure_output_structure(
                args.input, args.output, file_path
            )
            success, error = process_single_file(
                file_path, output_subdir, processors
            )
            if success:
                processed_count += 1
                processed_files.append(file_path)
            else:
                current_failed.append((file_path, error or "Unknown error"))
    else:
        with tqdm.tqdm(
            total=len(files),
            desc="Processing files",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
        ) as pbar:
            for file_path in files:
                output_subdir = ensure_output_structure(
                    args.input, args.output, file_path
                )
                success, error = process_single_file(
                    file_path, output_subdir, processors, pbar
                )
                if success:
                    processed_count += 1
                    processed_files.append(file_path)
                else:
                    current_failed.append(
                        (file_path, error or "Unknown error")
                    )
                pbar.update(1)

    if not args.no_state and (failed_files or current_failed):
        all_failed = failed_files + current_failed
        save_processing_state(args.state_file, processed_files, all_failed)

    usage_stats = collect_usage_stats(processors)
    if usage_stats:
        display_usage_stats(usage_stats)

    total_failed = len(current_failed)
    print(f"\nCompleted! Processed: {processed_count}, Failed: {total_failed}")
    print(f"Results saved to: {args.output}")

    if current_failed:
        print(f"\nFailed files:")
        for file_path, error in current_failed:
            print(f"  {file_path}: {error}")

        if not args.no_state:
            print(f"\nProcessing state saved to: {args.state_file}")
            print(
                "Run again to retry failed files or use --no-state to ignore state"
            )
