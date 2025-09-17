import os
import os.path as path
import shutil
from pathlib import Path
from subprocess import CalledProcessError, run
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
    path.join(
        glb.instructions_dir, f"describe_yolo_{YOLOScene.FIGURE.value}.txt"
    ),
    "r",
).read()
sys_msg_formula = open(
    path.join(
        glb.instructions_dir,
        f"describe_yolo_{YOLOScene.ISOLATE_FORMULA.value}.txt",
    ),
    "r",
).read()
sys_msg_text = open(
    path.join(
        glb.instructions_dir, f"describe_yolo_{YOLOScene.PLAIN_TEXT.value}.txt"
    ),
    "r",
).read()
sys_msg_table = open(
    path.join(
        glb.instructions_dir, f"describe_yolo_{YOLOScene.TABLE.value}.txt"
    ),
    "r",
).read()

yolo_sys_msgs = {
    YOLOScene.FIGURE: sys_msg_figure,
    YOLOScene.ISOLATE_FORMULA: sys_msg_formula,
    YOLOScene.PLAIN_TEXT: sys_msg_text,
    YOLOScene.TABLE: sys_msg_table,
}


def is_office_to_pdf_available() -> bool:
    if shutil.which("libreoffice") is not None:
        return True
    try:
        url = urljoin(cred.OFFICE_CONVERSION_SERVER, "/ping")
        return requests.get(url).status_code == 200
    except Exception:
        return False


def office_to_pdf(document_path: str, use_libre_cli: bool = False) -> str:
    temp_pdf_path = temp_file(".pdf")
    document_path_obj = Path(document_path)
    extension = document_path_obj.suffix.lower()

    if extension not in {".ppt", ".pptx", ".doc", ".docx"}:
        raise ValueError(f"Unsupported file format: {document_path}")

    if use_libre_cli:
        output_dir = Path(temp_pdf_path).parent
        try:
            run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(output_dir),
                    str(document_path_obj),
                ],
                check=True,
            )
            expected_pdf_path = output_dir / f"{document_path_obj.stem}.pdf"
            if not expected_pdf_path.exists():
                raise RuntimeError(f"PDF not generated for {document_path}")
            shutil.move(str(expected_pdf_path), temp_pdf_path)
            return temp_pdf_path
        except CalledProcessError as e:
            raise RuntimeError(f"LibreOffice conversion error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

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
