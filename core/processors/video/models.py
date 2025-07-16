from pydantic import BaseModel
from typing import Optional


class Span(BaseModel):
    start: float
    end: float
    short_description: Optional[str] = None
    text: str

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "short_description": self.short_description,
            "text": self.text,
        }


class Video(BaseModel):
    URI: str
    spans: list[Span]

    def to_json(self) -> dict:
        return {
            "URI": self.URI,
            "spans": [span.to_json() for span in self.spans],
        }
