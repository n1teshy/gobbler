import os.path as path
import random
import string
import tempfile


def temp_file(suffix: str, length: int = 10) -> str:
    letters = string.ascii_letters + string.digits
    random_name = "".join(random.choices(letters, k=length))
    return path.join(tempfile.gettempdir(), random_name + suffix)
