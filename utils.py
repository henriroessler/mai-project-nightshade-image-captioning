from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def rounded(self) -> BBox:
        x2 = self.x + self.w
        y2 = self.y + self.h
        ix1 = int(np.ceil(self.x))
        iy1 = int(np.ceil(self.y))
        ix2 = int(np.floor(x2))
        iy2 = int(np.floor(y2))
        return BBox(ix1, iy1, ix2 - ix1, iy2 - iy1)
