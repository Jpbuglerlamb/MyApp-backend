# core/rate_limit.py
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Bucket:
    tokens: float
    last: float

class TokenBucketLimiter:
    """
    Token bucket: allow 'rate' tokens per 'per_seconds' with burst up to 'capacity'.
    """
    def __init__(self, rate: float, per_seconds: float, capacity: float):
        self.rate = rate
        self.per_seconds = per_seconds
        self.capacity = capacity
        self._buckets: Dict[str, Bucket] = {}

    def allow(self, key: str, cost: float = 1.0) -> bool:
        now = time.time()
        b = self._buckets.get(key)
        if b is None:
            b = Bucket(tokens=self.capacity, last=now)
            self._buckets[key] = b

        # refill
        elapsed = now - b.last
        refill = (elapsed / self.per_seconds) * self.rate
        b.tokens = min(self.capacity, b.tokens + refill)
        b.last = now

        if b.tokens >= cost:
            b.tokens -= cost
            return True
        return False
