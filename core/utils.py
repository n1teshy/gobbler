import hashlib
import mimetypes
import os
import random
import string
import tempfile
from typing import Optional


def temp_file(suffix: str, length: int = 10) -> str:
    letters = string.ascii_letters + string.digits
    random_name = "".join(random.choices(letters, k=length))
    return os.path.join(tempfile.gettempdir(), random_name + suffix)


def hash_file(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_mime_type(file_path: str) -> Optional[str]:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type


def get_file_metadata(path: str, URI: Optional[str] = None) -> dict:
    mime = get_mime_type(path)
    if mime is None:
        raise ValueError(f"Unsupported file {path}")
    return dict(
        URI=URI or path,
        mime_type=mime,
        size=os.path.getsize(path),
        version=int(os.path.getmtime(path)),
        hash=hash_file(path),
    )
