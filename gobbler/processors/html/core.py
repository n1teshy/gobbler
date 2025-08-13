import os
import tempfile
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from trafilatura import extract

from gobbler.logger import logger
from gobbler.processors.image.core import ImageProcessor
from gobbler.processors.interfaces import BaseProcessor
from gobbler.processors.video.core import VideoProcessor
from gobbler.utils import (
    cleanup_temp_file,
    download,
    get_absolute_url,
    temp_file,
)


class HTMLProcessor(BaseProcessor):
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.video_processor = VideoProcessor()

    def get_media_file(self, src: str, base_url: str = "") -> Optional[str]:
        if os.path.exists(src):
            return src

        if base_url and os.path.isdir(base_url):
            file_path = os.path.join(base_url, src)
            if os.path.exists(file_path):
                return file_path

        absolute_url = get_absolute_url(src, base_url)
        if not absolute_url or absolute_url == src:
            return None

        try:
            success, path = download(absolute_url)
            return path if success else None
        except Exception as e:
            logger.warning(f"Failed to get HTML media {absolute_url}: {e}")
            return None

    def create_svg_file(self, svg_content: str) -> str:
        svg_path = temp_file(".svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return svg_path

    def is_base64_data_uri(self, src: str) -> bool:
        return src.startswith("data:") and ";base64," in src

    def decode_base64_media(self, data_uri: str) -> Optional[str]:
        try:
            header, encoded = data_uri.split(";base64,", 1)
            mime_type = header.replace("data:", "")

            import base64

            decoded_data = base64.b64decode(encoded)

            if mime_type.startswith("image/"):
                if "png" in mime_type:
                    suffix = ".png"
                elif "jpeg" in mime_type or "jpg" in mime_type:
                    suffix = ".jpg"
                elif "gif" in mime_type:
                    suffix = ".gif"
                elif "webp" in mime_type:
                    suffix = ".webp"
                else:
                    return None
            elif mime_type.startswith("video/"):
                if "mp4" in mime_type:
                    suffix = ".mp4"
                elif "webm" in mime_type:
                    suffix = ".webm"
                elif "avi" in mime_type:
                    suffix = ".avi"
                elif "mov" in mime_type or "quicktime" in mime_type:
                    suffix = ".mov"
                elif "mkv" in mime_type or "x-matroska" in mime_type:
                    suffix = ".mkv"
                elif "wmv" in mime_type or "x-ms-wmv" in mime_type:
                    suffix = ".wmv"
                elif "flv" in mime_type or "x-flv" in mime_type:
                    suffix = ".flv"
                else:
                    return None
            else:
                return None

            temp_path = temp_file(suffix)
            with open(temp_path, "wb") as f:
                f.write(decoded_data)
            return temp_path
        except Exception as e:
            logger.warning(f"Failed to decode base64 data URI: {e}")
            return None

    def process_media_element(self, tag, base_url: str = "") -> Optional[str]:
        try:
            src = tag.get("src")
            media_path, cleanup_after = None, False
            if not src:
                return None

            if self.is_base64_data_uri(src):
                media_path = self.decode_base64_media(src)
                cleanup_after = True
            else:
                media_path = self.get_media_file(src, base_url)

            if not media_path:
                return None

            try:
                if tag.name == "img":
                    processed = self.image_processor.process(media_path)
                    return self.format_media_tag("img", processed.to_json())
                elif tag.name == "video":
                    processed = self.video_processor.process(media_path)
                    if isinstance(processed, list):
                        return "\n".join(
                            [
                                self.format_media_tag("video", span.to_json())
                                for span in processed
                            ]
                        )
                    else:
                        return self.format_media_tag(
                            "video", processed.to_json()
                        )
            finally:
                if cleanup_after:
                    cleanup_temp_file(media_path)

        except Exception as e:
            logger.error(f"Failed to process {tag.name} element: {e}")
            return None

    def format_media_tag(self, tag_name: str, properties: dict) -> str:
        attrs = " ".join([f'{k}="{v}"' for k, v in properties.items()])
        return f"<{tag_name} {attrs}></{tag_name}>"

    def insert_media_in_text(self, text: str, media_map: dict) -> str:
        lines = text.split("\n")
        result_lines = []

        for line in lines:
            modified_line = line
            for placeholder, media_content in media_map.items():
                if placeholder in modified_line:
                    if media_content:
                        modified_line = modified_line.replace(
                            placeholder, media_content
                        )
                    else:
                        modified_line = modified_line.replace(placeholder, "")
            result_lines.append(modified_line)
        return "\n".join(result_lines)

    def process(self, path: str, base_url: str = "") -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(f"HTML file not found: {path}")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        media_elements = {}
        for idx, tag in enumerate(soup.find_all(["img", "video"])):
            placeholder = f"MEDIA_PLACEHOLDER_{tag.name}_{idx}"
            media_elements[placeholder] = self.process_media_element(
                tag, base_url
            )
            tag.replace_with(placeholder)

        text_content = extract(str(soup))
        if not text_content:
            text_content = soup.get_text()

        return self.insert_media_in_text(text_content, media_elements)
