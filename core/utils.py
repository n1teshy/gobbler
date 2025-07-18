import hashlib
import os.path as path
import random
import string
import tempfile


def temp_file(suffix: str, length: int = 10) -> str:
    letters = string.ascii_letters + string.digits
    random_name = "".join(random.choices(letters, k=length))
    return path.join(tempfile.gettempdir(), random_name + suffix)


def hash_file(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
