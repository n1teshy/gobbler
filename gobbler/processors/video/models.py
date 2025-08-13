import json
from typing import Union

from gobbler.processors.image.models import Image
from gobbler.processors.interfaces import BaseFile, DBEntity


class Span(DBEntity, BaseFile):
    start: float
    end: float
    short_description: str
    long_description: str
    frames: Union[dict[int, Image], list[Image]] = {}
    keywords: list[str] = []

    class Config:
        extra = "allow"

    def to_json(self) -> dict:
        return json.loads(super().model_dump_json())
