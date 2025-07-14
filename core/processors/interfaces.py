from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    type: str

    @abstractmethod
    def process(self, *args, **kwargs):
        pass
