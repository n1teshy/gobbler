import os
import os.path as path
from pathlib import Path
from urllib.parse import urljoin

import requests

import gobbler.constants as c
import gobbler.cred as cred
import gobbler.globals as glb
from gobbler.utils import temp_file

idx_2_chunk_type = [
    c.YOLO_TITLE,
    c.YOLO_PLAIN_TEXT,
    c.YOLO_FIGURE,
    c.YOLO_FIGURE_CAPTION,
    c.YOLO_TABLE,
    c.YOLO_TABLE_CAPTION,
    c.YOLO_TABLE_FOOTNOTE,
    c.YOLO_ISOLATE_FORMULA,
    c.YOLO_FORMULA_CAPTION,
]
chunk_type_2_idx = {typ: idx for idx, typ in enumerate(idx_2_chunk_type)}
conversion_endpoint = cred.OFFICE_CONVERSION_SERVER and urljoin(
    cred.OFFICE_CONVERSION_SERVER, "/convert"
)

sys_msg_any_caption = open(
    path.join(glb.instructions_dir, "describe_yolo_any_caption.txt"), "r"
).read()
sys_msg_figure = open(
    path.join(glb.instructions_dir, f"describe_yolo_{c.YOLO_FIGURE}.txt"), "r"
).read()
sys_msg_formula = open(
    path.join(
        glb.instructions_dir, f"describe_yolo_{c.YOLO_ISOLATE_FORMULA}.txt"
    ),
    "r",
).read()
sys_msg_text = open(
    path.join(glb.instructions_dir, f"describe_yolo_{c.YOLO_PLAIN_TEXT}.txt"),
    "r",
).read()
sys_msg_table = open(
    path.join(glb.instructions_dir, f"describe_yolo_{c.YOLO_TABLE}.txt"), "r"
).read()

yolo_sys_msgs = {
    c.YOLO_FIGURE: sys_msg_figure,
    c.YOLO_ISOLATE_FORMULA: sys_msg_formula,
    c.YOLO_PLAIN_TEXT: sys_msg_text,
    c.YOLO_TABLE: sys_msg_table,
}


def idx_to_chunk_type(idx: int) -> str:
    return idx_2_chunk_type[idx]


def chunk_type_to_idx(chunk_type: str) -> int:
    return chunk_type_2_idx[chunk_type]


def office_to_pdf(document_path: str) -> str:
    extension = Path(document_path).suffix.lower()
    if extension not in {".ppt", ".pptx", ".doc", ".docx"}:
        raise ValueError(f"Unsupported file format: {document_path}")

    temp_pdf_path = temp_file(".pdf")
    with open(document_path, "rb") as f:
        files = {
            "file": (
                os.path.basename(document_path),
                f,
                "application/octet-stream",
            )
        }
        if not conversion_endpoint:
            raise EnvironmentError("OFFICE_CONVERSION_URL is missing ")
        response = requests.post(conversion_endpoint, files=files)
        if response.status_code != 200:
            raise RuntimeError(f"Conversion failed: {response.text}")
        with open(temp_pdf_path, "wb") as pdf_out:
            pdf_out.write(response.content)
    return temp_pdf_path
