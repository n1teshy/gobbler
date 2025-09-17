# Gobbler Processors API Reference

<p align="center">
  <img src="https://dev.azure.com/Zifo/b0ba8dd6-79d2-4d03-8df4-f8407bc209de/_apis/git/repositories/5e6eeddf-d308-44a9-876e-1f4d3d6159b7/items?path=/assets/1.png&versionDescriptor%5BversionOptions%5D=0&versionDescriptor%5BversionType%5D=0&versionDescriptor%5Bversion%5D=dev/nitesh&resolveLfs=true&%24format=octetStream&api-version=5.0" alt="Poster Image" />
</p>


## Getting started

- Ensure [ffmpeg](https://ffmpeg.org/download.html) is installed (skip if you don't need video processing).

- If on windows, Build and run `Dockerfile.converter`, used to convert documents to PDF for easier processing (skip if you don't need non-PDF document processing).

  ```bash
  cd <project_directory>
  docker build -f Dockerfile.converter -t <image_name> .
  docker run -d -p 8000:8000 <image_name>
  ```

- If on linux, perform the last step or install [libreoffice](https://www.libreoffice.org/download/download-libreoffice/) and the necessary fonts, used to convert documents to PDF for easier processing (skip if you don't need non-PDF document processing).

  ```bash
  sudo apt-get update
  sudo apt-get install -y libreoffice fonts-noto fonts-noto-cjk fontconfig fonts-freefont-ttf
  ```

- Install Gobbler.

  ```bash
  pip install "git+https://dev.azure.com/Zifo/AIdeate%20and%20AIterate/_git/Multi-Modal%20Data%20Ingestion%20Pipeline"
  ```

  > NOTE: the code may not have been merged to main branch, try "git+https://dev.azure.com/Zifo/AIdeate%20and%20AIterate/_git/Multi-Modal%20Data%20Ingestion%20Pipeline@dev/nitesh" in case installation fails

- Ensure all environment variables are set, read `env_instructions.txt` for more info.

## CLI
Try using the cli for quicker runs.
```bash
python -m gobbler -h
````

## Quick Usage Snippets
Minimal examples showing instantiation and calling `process()`.

```python
# Document processing (PDF or Office -> PDF)
from gobbler.processors.docs.core import DocumentProcessor

doc_proc = DocumentProcessor()
objects = doc_proc.process("/path/to/file.docx", use_fitz_on_text=True)
for obj in objects:
    print(obj.page, obj.type, obj.keywords)
```

```python
# Image processing
from gobbler.processors.image.core import ImageProcessor
img_proc = ImageProcessor()
img_obj = img_proc.process("/path/to/image.png")
print(img_obj.scene, img_obj.description)
```

```python
# Video processing (combined audio + frames)
from gobbler.processors.video.core import VideoProcessor
vid_proc = VideoProcessor(spf=5)  # one frame every 5 seconds
spans = vid_proc.process("/path/to/video.mp4")
for span in spans:
    print(f"{span.start:.1f}-{span.end:.1f}", span.short_description)
```

```python
# Video frames only (no transcription)
frames, ranges = vid_proc.process("/path/to/video.mp4", frames_only=True)
print(len(frames), "frames extracted")
```

```python
# HTML processing (enrich text with media descriptions)
from gobbler.processors.html.core import HTMLProcessor
html_proc = HTMLProcessor()
text_with_media = html_proc.process("/path/to/page.html")
print(text_with_media[:500])
```

```python
# Custom prompts example (image)
from gobbler.models.utils import ClipScene
custom = {ClipScene.DIAGRAM: "Describe the diagram focusing on axes and trends."}
img_obj = img_proc.process("/path/to/diagram.png", scene_to_prompt=custom, fallback_prompt="Describe the content.")
```

> NOTE: Azure/OpenAI credentials, LibreOffice, ffmpeg, etc. must be configured per environment variables referenced in `gobbler.cred` before these snippets succeed.

## DocumentProcessor.process()
**Location:** `gobbler/processors/docs/core.py`

Extracts structured objects (figures, tables, text regions, formulas, etc.) from a document (PDF or Office file). Non-PDF Office files are first converted to PDF using LibreOffice (local) or a remote conversion service.

Parameters:
- `path: str` Absolute or relative path to the input file. If not found, raises `FileNotFoundError`.
- `no_ocr: bool = False` If True, skips OCR / vision model calls; descriptions will be empty strings.
- `use_fitz_on_text: bool = False` If True, for text-like YOLO detections (plain text, titles, *_caption) it extracts text directly with PyMuPDF instead of OCR.
- `yolo_class_to_prompt: Optional[dict[YOLOScene, str]] = None` Custom mapping of YOLO object classes to system prompts. Must be provided together with `yolo_fallback_prompt`.
- `yolo_fallback_prompt: Optional[str] = None` Fallback system prompt when a class-specific prompt is absent. Must be provided with `yolo_class_to_prompt`.
- `identify_keywords: bool = True` If True, keywords are extracted (via KeyBERT) either from raw description text (when prompts are user supplied or using fitz text mode) or taken from model JSON output.
- `only_pages: list[int] = []` list of 0-indexed page numbers that will be processed, use this if you want to skip some pages.
- `use_libre_cli: bool = False` Specifies whether LibreOffice CLI (local) should to convert documents to PDF, otherwise the document conversion server is used.

Return Type:
- `list[DocumentObject]`
  - Each `DocumentObject` fields:
    - `id: int | None` (database id placeholder, may be None)
    - `uri: str` original file URI/path
    - `mime_type: str`
    - `size: int` bytes
    - `uploaded_by: str`
    - `uploaded_at: int` epoch seconds
    - `version: int`
    - `hash: str` content hash
    - `page: int` zero-based page index
    - `position: Position(x1, y1, x2, y2)` page pixel (or point) rectangle
    - `type: str | None` detected scene / object label (e.g., figure, table, plain_text)
    - `content: str` textual description or extracted text; for tables may contain HTML table snippet
    - `keywords: list[str]` keyword list (may be empty)
  - Convenience property:
    - `table -> list[list[Any]] | [] | None` parsed 2D table rows (None if not a table; empty list if parsing failed)

Behavior:
1. Optionally converts non-PDF Office docs to PDF.
2. Opens the PDF with PyMuPDF and iterates pages.
3. Runs YOLO to detect regions; skips `ABANDON` class.
4. For each region: either crops and OCRs / describes via vision model or pulls text directly (if `use_fitz_on_text`).
5. Gathers keywords either by KeyBERT on text content or from JSON returned by the model.
6. Returns `list[DocumentObject]` with positional metadata, type, content, and keywords.

Raises:
- `AssertionError` if only one of `yolo_class_to_prompt` / `yolo_fallback_prompt` is supplied.
- `FileNotFoundError` if `path` does not exist.
- Propagates runtime errors from conversion or downstream model usage.

---

## ImageProcessor.process()
**Location:** `gobbler/processors/image/core.py`

Classifies and (optionally) OCRs / describes an image, returning a structured `Image` object.

Parameters:
- `path: str` Path to an image file (must have MIME type starting with `image/`).
- `no_ocr: bool = False` If True, skips model call and returns empty description & keywords.
- `scene: Optional[ClipScene] = None` Force a scene classification instead of automatic CLIP-based classification.
- `scene_to_prompt: Optional[dict[ClipScene, str]] = None` Custom system prompts per scene; must be given together with `fallback_prompt`.
- `fallback_prompt: Optional[str] = None` Fallback system prompt; must accompany `scene_to_prompt`.
- `identify_keywords: bool = True` If True, extract keywords (KeyBERT) when custom prompts are provided; otherwise rely on model JSON.

Return Type:
- `Image`
  - Fields:
    - `id: int | None`
    - `uri: str`
    - `mime_type: str`
    - `size: int`
    - `uploaded_by: str`
    - `uploaded_at: int`
    - `version: int`
    - `hash: str`
    - `shape: str` "<height>x<width>"
    - `scene: ClipScene | None` inferred or provided scene (TEXT, DIAGRAM, TABULAR, etc.)
    - `description: str`
    - `keywords: list[str]`

Behavior:
1. Validates file existence and MIME type.
2. Classifies scene (unless provided) using CLIP plus heuristic adjustments.
3. If `no_ocr` returns immediately with blank description & keywords.
4. Builds prompt mapping (default or user provided) and selects fallback.
5. Calls Azure Vision model (chat completions) to obtain description + keywords (direct JSON or plain text + post-keyword extraction depending on user prompts).
6. Returns `Image` with metadata, shape, scene, description, and keywords.

Raises:
- `AssertionError` for mismatched prompt args.
- `FileNotFoundError` if path missing.
- `ValueError` if MIME type not an image.
- `RuntimeError` if description generation fails.

---

## VideoProcessor.process()
**Location:** `gobbler/processors/video/core.py`

Extracts temporal topic spans (and optionally key frames / frame analyses) from a video. Can operate in three modes: audio-only, frames-only, or combined.

Parameters:
- `path: str` Video file path.
- `no_ocr: bool = False` If True, frame OCR/description is skipped; frames will still be sampled and classified if needed, but descriptions & keywords will be blank.
- `audio_only: bool = False` If True, only audio transcription + topic segmentation is performed (no frame extraction/assignment).
- `frames_only: bool = False` If True, only returns processed frame images + time ranges (no audio transcription/topics). Automatically enforced if the video has no audio track (unless `audio_only=True`, which then raises).
- `frame_scene_to_prompt: Optional[dict[ClipScene, str]] = None` Custom prompts for frame OCR; must accompany `frame_fallback_prompt`.
- `frame_fallback_prompt: Optional[str] = None` Fallback prompt for frames; must accompany `frame_scene_to_prompt`.
- `identify_keywords: bool = True` If True, extracts keywords (similar logic to ImageProcessor) for frame descriptions.

Return Types:
- Combined or audio modes: `list[Span]`
  - Each `Span` fields:
    - `id: int | None`
    - `uri: str`
    - `mime_type: str`
    - `size: int`
    - `uploaded_by: str`
    - `uploaded_at: int`
    - `version: int`
    - `hash: str`
    - `start: float` seconds
    - `end: float` seconds
    - `short_description: str` summary for the time range
    - `long_description: str` concatenated transcript text in the range
    - `keywords: list[str]`
    - `frames: dict[int, Image] | list[Image]` mapping midpoint timestamps to key frame `Image` objects (dict form used here). Empty if `audio_only`.
- Frames-only mode: `tuple[list[Image], list[dict[str, float]]]`
  - First element: list of processed frame `Image` objects (see Image fields above)
  - Second element: list of dicts `{ "start": float, "end": float }` giving temporal coverage for each frame

Behavior:
1. Validates file existence and video MIME type.
2. Determines if audio track exists; adjusts mode (forces frames-only if no audio and not frames_only; errors if audio_only is set without audio).
3. If frames-only: extracts frames (skip & dedupe via SSIM / histogram), processes each with `ImageProcessor`, and returns frames + their time ranges.
4. Otherwise: transcribes audio (Azure Whisper), segments into topics (Azure LLM with structured JSON), builds spans.
5. If not `audio_only`, also processes frames then assigns frames to spans when overlap >= configured seconds-per-frame.
6. Returns final spans or frame tuple depending on mode.

Raises:
- `AssertionError` for mismatched prompt args.
- `FileNotFoundError` if path missing.
- `ValueError` for non-video input or invalid audio-only request without audio track.
- `RuntimeError` if topic extraction fails.

---

## HTMLProcessor.process()
**Location:** `gobbler/processors/html/core.py`

Parses an HTML file, extracts textual content, and inlines processed representations of embedded images and videos (optionally restricted to a subset of media types).

Parameters:
- `path: str` Path to an HTML file.
- `base_url: str = ""` Base directory or URL used to resolve relative media `src` paths.
- `no_ocr: bool = None` If truthy, skips media OCR and returns only extracted plain text (trafilatura or fallback DOM text).
- `img_only: bool = False` If True, only `<img>` tags are processed; videos ignored.
- `video_only: bool = False` If True, only `<video>` tags are processed; images ignored.

Return Type:
- `str` enriched plain text. Media references replaced with synthetic tags of the form `<img key="..." ... ></img>` or `<video key="..." ... ></video>` containing serialized metadata from underlying `Image` / `Span.to_json()` outputs. When multiple video spans returned for a `<video>` tag they are concatenated line-wise.

Behavior:
1. Loads and parses HTML with BeautifulSoup.
2. If `no_ocr`, returns extracted plain text only.
3. Determines allowed media tags based on `img_only` / `video_only`.
4. For each allowed media element: resolves or downloads the media file (including base64 data URIs), invokes `ImageProcessor` or `VideoProcessor`, and replaces the element with a placeholder.
5. Extracts main textual content with `trafilatura` (fallback: raw text) and replaces placeholders with serialized media tag representations.
6. Returns a single string containing enriched text with synthetic media tags.

Raises:
- `FileNotFoundError` if HTML path missing.
- Logs and skips media elements that fail processing (does not raise unless file missing).

---

## Office Conversion (Helper)
Although not a `process()` method, document ingestion depends on `office_to_pdf()` in `gobbler/processors/docs/utils.py`, which performs local LibreOffice conversion (preferred) or falls back to a remote service.

---

## Common Patterns & Conventions
- All processors validate file existence early and raise `FileNotFoundError` on missing input.
- Paired prompt arguments must be provided together (asserted) to avoid partial customization.
- Keyword extraction logic depends on whether user prompts (plain text responses) or model JSON responses are used.
- Temporary files (converted PDFs, extracted frames, audio chunks) are cleaned up where possible; warnings are logged if cleanup fails.
