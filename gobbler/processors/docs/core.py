import os
import warnings
from urllib.parse import urljoin

import fitz
import requests
from agentic_doc.parse import parse
from keybert import KeyBERT

import gobbler.constants as c
import gobbler.cred as cred
from gobbler.logger import logger
from gobbler.processors.docs.models import DocumentObject, Position
from gobbler.processors.docs.utils import office_to_pdf
from gobbler.processors.interfaces import BaseProcessor
from gobbler.utils import (
    dump_usage_data,
    get_file_metadata,
    get_usage_file,
    load_usage_data,
)


class DocumentProcessor(BaseProcessor):
    def __init__(self):
        if cred.LANDINGAI_KEY is None:
            raise EnvironmentError("LandingAI key is missing")
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
        self.usage_file = get_usage_file(c.USAGE_LAI_OCR)
        self.usage_data = load_usage_data(self.usage_file)
        self.usage_data["pages"] = self.usage_data.get("pages", 0)

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
                img_bytes = page.get_pixmap().tobytes()
                objects = parse(img_bytes)[0].chunks
                for o in objects:
                    box = o.grounding[0].box
                    doc_objects.append(
                        DocumentObject(
                            **metadata,
                            page=page_idx,
                            position=Position(box.t, box.r, box.b, box.l),
                            type=o.chunk_type.value,
                            content=o.text,
                            keywords=list(
                                map(
                                    lambda x: x[0],
                                    self.keybert.extract_keywords(o.text),
                                )
                            ),
                        )
                    )
                self.usage_data["pages"] += 1
                dump_usage_data(self.usage_data, self.usage_file)
            return doc_objects
        finally:
            try:
                fitz_doc and fitz_doc.close()
                converted_pdf and os.remove(converted_pdf)
            except PermissionError as e:
                logger.warning(f"Failed to delete temporary file: {e}")
