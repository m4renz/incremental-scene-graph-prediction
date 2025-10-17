from __future__ import annotations
import dataclasses
from numbers import Number

@dataclasses.dataclass(slots=True)
class BoundingBox2D:
    """A very minimal 2D bounding box implementation.
    """
    x_min: Number = 0
    y_min: Number = 0
    x_max: Number = 0
    y_max: Number = 0

    def valid(self) -> bool:
        return self.x_max > self.x_min and self.y_max > self.y_min
    
    def tolist(self):
        return [getattr(self, field.name) for field in dataclasses.fields(self)]

    def size(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def shape(self):
        return self.x_max - self.x_min, self.y_max - self.y_min

    def normalize(self, shape: tuple[int, int]) -> None:
        self.x_min /= shape[0]
        self.y_min /= shape[1]
        self.x_max /= shape[0]
        self.y_max /= shape[1]

    def denormalize(self, shape: tuple[int, int]) -> None:
        self.x_min *= shape[0]
        self.y_min *= shape[1]
        self.x_max *= shape[0]
        self.y_max *= shape[1]

    def __getitem__(self, index):
        field = dataclasses.fields(self)[index]
        return getattr(self, field.name)