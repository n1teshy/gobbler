from typing import Optional

from pydantic import BaseModel

from core.processors.image.models import Image


class Span(BaseModel):
    start: float
    end: float
    short_description: Optional[str] = None
    text: str
    frames: list[Image] = []

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "short_description": self.short_description,
            "text": self.text,
            "frames": [frame.to_json() for frame in self.frames],
        }


class Video(BaseModel):
    URI: str
    spans: list[Span]

    def to_json(self) -> dict:
        return {
            "URI": self.URI,
            "spans": [span.to_json() for span in self.spans],
        }
