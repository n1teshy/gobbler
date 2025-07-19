from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel, Field


class BaseProcessor(ABC):
    type: str

    @abstractmethod
    def process(self, *args, **kwargs):
        pass


class BaseFile(BaseModel):
    URI: str
    mime_type: str
    size: int
    uploaded_by: str = "system"
    uploaded_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    version: float
    hash: str

    def to_json(self) -> dict:
        return {
            "URI": self.URI,
            "mime_type": self.mime_type,
            "size": self.size,
            "uploaded_by": self.uploaded_by,
            "uploaded_at": self.uploaded_at,
            "version": self.version,
            "hash": self.hash,
        }
