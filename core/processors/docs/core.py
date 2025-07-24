import os

import fitz
from agentic_doc.parse import parse
from keybert import KeyBERT

import core.cred as cred
from core.processors.docs.models import DocumentObject, Position
from core.processors.docs.utils import office_to_pdf
from core.processors.interfaces import BaseProcessor
from core.utils import get_file_metadata


class DocumentProcessor(BaseProcessor):
    def __init__(self):
        if cred.LANDINGAI_KEY is None:
            raise EnvironmentError("LandingAI key is missing")
        self.keybert = KeyBERT("bert-base-nli-mean-tokens")

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
            return doc_objects
        finally:
            try:
                fitz_doc and fitz_doc.close()
                converted_pdf and os.remove(converted_pdf)
            except PermissionError as e:
                print(f"Failed to delete temporary file: {e}")
