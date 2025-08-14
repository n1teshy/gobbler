import os
import os.path as path
from pathlib import Path
from urllib.parse import urljoin

import requests

import gobbler.cred as cred
import gobbler.globals as glb
from gobbler.models.utils import YOLOScene
from gobbler.utils import temp_file

conversion_endpoint = cred.OFFICE_CONVERSION_SERVER and urljoin(
    cred.OFFICE_CONVERSION_SERVER, "/convert"
)

sys_msg_any_caption = open(
    path.join(glb.instructions_dir, "describe_yolo_any_caption.txt"), "r"
).read()
sys_msg_figure = open(
    path.join(glb.instructions_dir, f"describe_yolo_{YOLOScene.FIGURE}.txt"),
    "r",
).read()
sys_msg_formula = open(
    path.join(
        glb.instructions_dir, f"describe_yolo_{YOLOScene.ISOLATE_FORMULA}.txt"
    ),
    "r",
).read()
sys_msg_text = open(
    path.join(
        glb.instructions_dir, f"describe_yolo_{YOLOScene.PLAIN_TEXT}.txt"
    ),
    "r",
).read()
sys_msg_table = open(
    path.join(glb.instructions_dir, f"describe_yolo_{YOLOScene.TABLE}.txt"),
    "r",
).read()

yolo_sys_msgs = {
    YOLOScene.FIGURE: sys_msg_figure,
    YOLOScene.ISOLATE_FORMULA: sys_msg_formula,
    YOLOScene.PLAIN_TEXT: sys_msg_text,
    YOLOScene.TABLE: sys_msg_table,
}


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
