import io
import json
import os
from typing import Optional

import fitz

import gobbler.cred as cred
from gobbler.logger import logger
from gobbler.models.core import run_keybert, run_yolo
from gobbler.models.utils import YOLOScene
from gobbler.processors.docs.models import DocumentObject, Position
from gobbler.processors.docs.utils import (
    is_office_to_pdf_available,
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
    this_or_that,
)


class DocumentProcessor(BaseProcessor):
    def __init__(self):
        if not is_office_to_pdf_available():
            logger.warning(
                "libreoffice is not available, will not be able to process non-PDF files"
            )
        self.image_processor = ImageProcessor()

    def process_page(
        self,
        page: fitz.Page,
        no_ocr: bool,
        use_fitz: bool,
        yolo_class_to_prompt: dict[YOLOScene, str],
        yolo_fallback_prompt: str,
        prompts_from_user: bool,
    ) -> list[tuple[Position, YOLOScene, str]]:
        """
        Returns tuple[position, object_type, description]
        see constants file for YOLO objects.
        """
        img_bytes = page.get_pixmap().tobytes()
        page_image = make_pil_images([io.BytesIO(img_bytes)])[0]
        bboxes = run_yolo([page_image])[0]
        results = []

        for *coord, scene, _ in bboxes:
            if scene == YOLOScene.ABANDON:
                continue
            if scene is None:
                box_image = page_image
            else:
                box_image = page_image.crop(coord)
            if use_fitz and (
                scene in (YOLOScene.PLAIN_TEXT, YOLOScene.TITLE)
                or scene.value.endswith("_caption")
            ):
                description = page.get_textbox(fitz.Rect(*coord))
            elif not no_ocr:
                sys_msg = yolo_class_to_prompt.get(scene, yolo_fallback_prompt)
                description = self.image_processor.call_4o(
                    sys_msg,
                    stringify_image(box_image),
                    response_format=(
                        "text" if prompts_from_user else "json_object"
                    ),
                )
            else:
                description = ""
            results.append((Position(*coord), scene, description))

        return results

    def process(
        self,
        path: str,
        no_ocr: bool = False,
        use_fitz_on_text: bool = False,
        yolo_class_to_prompt: Optional[dict[YOLOScene, str]] = None,
        yolo_fallback_prompt: Optional[str] = None,
        identify_keywords: bool = True,
    ) -> list[DocumentObject]:
        assert (
            yolo_class_to_prompt is None == yolo_fallback_prompt is None
        ), "Either pass both 'yolo_class_to_prompt' and 'yolo_fallback_prompt' or none"

        prompts_from_user = yolo_fallback_prompt is not None
        yolo_class_to_prompt = this_or_that(
            yolo_class_to_prompt, yolo_sys_msgs
        )
        yolo_fallback_prompt = this_or_that(
            yolo_fallback_prompt, sys_msg_any_caption
        )
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
                page_boxes = self.process_page(
                    page,
                    no_ocr,
                    use_fitz_on_text,
                    yolo_class_to_prompt,
                    yolo_fallback_prompt,
                    prompts_from_user,
                )
                for position, label, description in page_boxes:
                    keywords = []
                    if identify_keywords and (
                        use_fitz_on_text or prompts_from_user
                    ):
                        keywords = [
                            word for word, _ in run_keybert(description)
                        ]
                    else:
                        desc_obj = json.loads(description)
                        description, keywords = (
                            desc_obj["description"],
                            desc_obj["keywords"],
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
