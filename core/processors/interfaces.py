from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BaseProcessor(ABC):
    type: str

    @abstractmethod
    def process(self, *args, **kwargs):
        pass


class DBEntity(BaseModel):
    id: Optional[int] = None

    def to_json(self) -> dict:
        return {"id": self.id}


class BaseFile(BaseModel):
    URI: str
    mime_type: str
    size: int
    uploaded_by: str = "system"
    uploaded_at: float = Field(default_factory=lambda: int(datetime.now().timestamp()))
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
