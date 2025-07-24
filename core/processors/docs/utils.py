import os
from pathlib import Path

import comtypes.client

from core.utils import temp_file


def office_to_pdf(document_path: str) -> str:
    POWERPOINT_FORMATS = {".ppt", ".pptx"}
    WORD_FORMATS = {".doc", ".docx"}

    extension = Path(document_path).suffix.lower()
    if extension in POWERPOINT_FORMATS:
        file_format = "powerpoint"
    elif extension in WORD_FORMATS:
        file_format = "word"
    else:
        raise ValueError(f"Unsupported file format: {document_path}")

    temp_pdf_path = temp_file(".pdf")
    app = None
    doc = None
    try:
        if file_format == "powerpoint":
            app = comtypes.client.CreateObject("Powerpoint.Application")
            app.Visible = True
            doc = app.Presentations.Open(
                os.path.abspath(document_path), WithWindow=False
            )
            doc.SaveAs(os.path.abspath(temp_pdf_path), 32)
        elif file_format == "word":
            app = comtypes.client.CreateObject("Word.Application")
            app.Visible = False
            doc = app.Documents.Open(os.path.abspath(document_path))
            doc.SaveAs2(os.path.abspath(temp_pdf_path), 17)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
        return temp_pdf_path
    finally:
        try:
            if doc:
                doc.Close()
            if app:
                app.Quit()
            del app, doc
        except:
            pass
