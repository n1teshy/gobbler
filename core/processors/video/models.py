from pydantic import BaseModel


class Span(BaseModel):
    start: float
    end: float
    text: str

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
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
