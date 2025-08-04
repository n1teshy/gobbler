import hashlib
import json
import mimetypes
import os
import random
import string
import tempfile
import time
from typing import Optional
from urllib.parse import urlparse

import appdirs
import requests

import gobbler.meta as meta


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


def get_file_metadata(path: str, uri: Optional[str] = None) -> dict:
    mime = get_mime_type(path)
    if mime is None:
        raise ValueError(f"Unsupported file {path}")
    return dict(
        uri=uri or path,
        mime_type=mime,
        size=os.path.getsize(path),
        version=int(os.path.getmtime(path)),
        hash=hash_file(path),
    )


def get_usage_file(usage_type: str) -> str:
    # one file per day
    today = int(time.time()) // 86400
    dir = os.path.join(appdirs.user_data_dir(meta.name), usage_type)
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, f"{today}.json")


def load_usage_data(file: str) -> dict:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            json.dump({}, f)
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_usage_data(data: dict, file: str) -> None:
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def download(
    url: str,
    path: Optional[str] = None,
    headers: Optional[dict] = None,
    chunk_size: int = 8192,
) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "file", ""}:
        raise ValueError(f"Unsupported URI scheme: {url}")
    if parsed.scheme in {"http", "https"}:
        path = path or temp_file(suffix=os.path.splitext(parsed.path)[-1])
        with requests.get(url, stream=True, headers=headers) as response:
            response.raise_for_status()
            with open(path, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        out_file.write(chunk)
        return True, path
    return False, url
