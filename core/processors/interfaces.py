from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

import core.constants as c


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
    uri: str
    mime_type: str
    size: int
    uploaded_by: str = "system"
    uploaded_at: int = Field(
        default_factory=lambda: int(datetime.now().timestamp())
    )
    version: int
    hash: str

    def to_json(self) -> dict:
        return {
            c.DB_FLD_URI: self.uri,
            c.DB_FLD_MIME_TYPE: self.mime_type,
            c.DB_FLD_SIZE: self.size,
            c.DB_FLD_UPLOADED_BY: self.uploaded_by,
            c.DB_FLD_UPLOADED_AT: self.uploaded_at,
            c.DB_FLD_VERSION: self.version,
            c.DB_FLD_HASH: self.hash,
        }
