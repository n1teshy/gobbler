from core.processors.interfaces import BaseProcessor
from core.processors.image.utils import classify_images
from typing import Optional


class ImageProcessor(BaseProcessor):
    def __init__(self, path: str):
        self.path = path
        self.description: Optional[str] = None

    def classify(self) -> int:
        return classify_images([self.path])[0]

    def describe(self) -> str:
        if self.description is None:
            self.description = ...
        return self.description
