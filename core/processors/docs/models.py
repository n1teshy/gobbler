from collections import namedtuple

from core.processors.interfaces import BaseFile, DBEntity

Position = namedtuple("Position", ["top", "right", "bottom", "left"])


class DocumentObject(DBEntity, BaseFile):
    page: int
    position: Position
    type: str
    content: str
    keywords: list[str] = []

    def to_json(self) -> dict:
        return {
            **DBEntity.to_json(self),
            **BaseFile.to_json(self),
            "position": list(self.position),
            "type": self.type,
            "content": self.content,
            "keywords": self.keywords,
        }
