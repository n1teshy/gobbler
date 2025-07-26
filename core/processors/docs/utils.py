import os
from pathlib import Path

import requests
from agentic_doc.common import ChunkType

import core.cred as cred
from core.utils import temp_file

idx_2_chunk_type = [
    ChunkType.text,
    ChunkType.table,
    ChunkType.figure,
    ChunkType.marginalia,
]
chunk_type_2_idx = {typ: idx for idx, typ in enumerate(idx_2_chunk_type)}


def idx_to_chunk_type(idx: int) -> ChunkType:
    return idx_2_chunk_type[idx]


def chunk_type_to_idx(chunk_type: ChunkType) -> int:
    return chunk_type_2_idx[chunk_type]


def office_to_pdf(document_path: str) -> str:
    extension = Path(document_path).suffix.lower()
    if extension not in {".ppt", ".pptx", ".doc", ".docx"}:
        raise ValueError(f"Unsupported file format: {document_path}")

    temp_pdf_path = temp_file(".pdf")
    with open(document_path, "rb") as f:
        files = {
            "file": (os.path.basename(document_path), f, "application/octet-stream")
        }
        if cred.OFFICE_CONVERSION_URL is None:
            raise EnvironmentError("OFFICE_CONVERSION_URL is missing ")
        response = requests.post(cred.OFFICE_CONVERSION_URL, files=files)
        if response.status_code != 200:
            raise RuntimeError(f"Conversion failed: {response.text}")
        with open(temp_pdf_path, "wb") as pdf_out:
            pdf_out.write(response.content)
    return temp_pdf_path
