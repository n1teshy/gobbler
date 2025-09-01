from io import BytesIO
from logging import INFO, FileHandler, Formatter, getLogger
from pathlib import Path
from subprocess import CalledProcessError, run
from tempfile import NamedTemporaryFile
from typing import Any

try:
    from fastapi import FastAPI, File, Response, UploadFile
    from fastapi.responses import StreamingResponse
except ImportError as e:
    print(f"missing dependencies! {e}")
    print("pip install fastapi uvicorn python-multipart")

app = FastAPI()
logger = getLogger("convert-logger")
handler = FileHandler("app.log")
handler.setFormatter(Formatter("%(levelname)s:%(asctime)s:%(message)s"))
logger.addHandler(handler)
logger.setLevel(INFO)


def convert_to_pdf(input_bytes: bytes, input_filename: str) -> bytes:
    with NamedTemporaryFile(
        suffix=Path(input_filename).suffix, delete=False
    ) as input_file:
        input_file.write(input_bytes)
        input_file.flush()
        input_path = Path(input_file.name)
    pdf_path = input_path.with_suffix(".pdf")
    pdf_bytes = b""
    try:
        result = run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(input_path.parent),
                str(input_path),
            ],
            capture_output=True,
            check=True,
        )
        logger.info(
            f"convert: filename={input_filename}, returncode={result.returncode}, stderr={result.stderr.decode()}, stdout={result.stdout.decode()}"
        )
        if pdf_path.exists():
            pdf_bytes = pdf_path.read_bytes()
        else:
            logger.error(f"PDF not generated for {input_filename}")
    except CalledProcessError as e:
        logger.error(
            f"LibreOffice conversion error: {e}, stderr: {e.stderr.decode()}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if input_path.exists():
            input_path.unlink(missing_ok=True)
        if pdf_path.exists():
            pdf_path.unlink(missing_ok=True)
    return pdf_bytes


@app.get("/ping")
def ping() -> Response:
    return Response("pong", status_code=200)


@app.post("/convert")
async def convert(file: UploadFile = File(...)) -> Any:
    if not file.filename.lower().endswith((".doc", ".docx", ".ppt", ".pptx")):
        logger.warning(f"invalid file: {file.filename}")
        return Response("Invalid file type", status_code=400)
    input_bytes = await file.read()
    pdf_bytes = convert_to_pdf(input_bytes, file.filename)
    if not pdf_bytes:
        logger.error(f"conversion failed: {file.filename}")
        return Response("Conversion failed", status_code=500)
    return StreamingResponse(BytesIO(pdf_bytes), media_type="application/pdf")
