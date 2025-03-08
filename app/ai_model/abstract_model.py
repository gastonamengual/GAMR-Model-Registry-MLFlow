from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class AbstractModel(ABC):
    @abstractmethod
    def train(X, y): ...