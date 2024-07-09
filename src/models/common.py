from dataclasses import dataclass


@dataclass
class RangeWeight:
    start: int
    end: int
    weight: float
