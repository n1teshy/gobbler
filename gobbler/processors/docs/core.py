import io
import os
from urllib.parse import urljoin

import fitz
import requests
from keybert import KeyBERT

import gobbler.constants as c
import gobbler.cred as cred
import gobbler.globals as glb
from gobbler.logger import logger
from gobbler.models.core import run_yolo
from gobbler.processors.docs.models import DocumentObject, Position
from gobbler.processors.docs.utils import (
    office_to_pdf,
    sys_msg_any_caption,
    yolo_sys_msgs,
)
from gobbler.processors.image.core import ImageProcessor
from gobbler.processors.interfaces import BaseProcessor
from gobbler.utils import (
    get_file_metadata,
    make_pil_images,
    stringify_image,
)


class DocumentProcessor(BaseProcessor):
    def __init__(self):
        if cred.OFFICE_CONVERSION_SERVER is None:
            logger.warning(
                "OFFICE_CONVERSION_URL is not set, you'll have a hard time with non-PDF files"
            )
        else:
            try:
                url = urljoin(cred.OFFICE_CONVERSION_SERVER, "/ping")
                assert requests.get(url).status_code == 200
            except Exception as e:
                logger.warning(
                    f"Office conversion server is not reachable: {e}"
                )

        self.keybert = KeyBERT("bert-base-nli-mean-tokens")
        self.image_processor = ImageProcessor()

    def process_page(self, page: fitz.Page) -> list[tuple[Position, str, str]]:
        """
        Returns tuple[position, object_type, description]
        see constants file for YOLO objects.
        """
        img_bytes = page.get_pixmap().tobytes()
        page_image = make_pil_images([io.BytesIO(img_bytes)])[0]
        bboxes = run_yolo([page_image])[0]
        results = []

        for *coord, label, _ in bboxes:
            if label == c.YOLO_ABANDON:
                continue
            if label is None:
                box_image = page_image
            else:
                box_image = page_image.crop(coord)
            sys_msg = yolo_sys_msgs.get(label, sys_msg_any_caption)
            if glb.no_caption_mode:
                description = ""  # Placeholder for batch processing
            else:
                description = self.image_processor.call_4o(
                    sys_msg, stringify_image(box_image), response_format="text"
                )
            results.append((Position(*coord), label, description))

        return results

    def process(self, path: str) -> list[DocumentObject]:
        metadata = get_file_metadata(path)
        doc_objects = []
        fitz_doc = None
        converted_pdf = None

        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
            if os.path.splitext(path)[1] != ".pdf":
                converted_pdf = office_to_pdf(path)

            fitz_doc = fitz.open(converted_pdf or path)
            for page_idx, page in enumerate(fitz_doc):
                page_boxes = self.process_page(page)
                for position, label, description in page_boxes:
                    if glb.no_caption_mode or not description:
                        keywords = []
                    else:
                        keywords = list(
                            map(
                                lambda x: x[0],
                                self.keybert.extract_keywords(description),
                            )
                        )

                    doc_objects.append(
                        DocumentObject(
                            **metadata,
                            page=page_idx,
                            position=position,
                            type=label,
                            content=description,
                            keywords=keywords,
                        )
                    )
            return doc_objects
        finally:
            try:
                fitz_doc and fitz_doc.close()
                converted_pdf and os.remove(converted_pdf)
            except PermissionError as e:
                logger.warning(f"Failed to delete temporary file: {e}")
