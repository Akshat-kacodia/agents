from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from event import InterruptionEvent
from filler import FillerClassifier

logger = logging.getLogger(__name__)

class ContextWindow:
    """
    Simple rolling buffer of recent speech segments.
    Used for metadata and stats rather than hard decisions.
    """

    def __init__(self, max_events: int = 12) -> None:
        self._events: Deque[Tuple[datetime, str, float]] = deque(maxlen=max_events)

    def add(self, transcript: str, confidence: float, timestamp: datetime) -> None:
        self._events.append((timestamp, transcript, confidence))

    def snapshot(self) -> Dict[str, float]:
        if not self._events:
            return {
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "confidence_trend": 0.0,
                "avg_length": 0.0,
            }

        confidences = [c for (_, _, c) in self._events]
        lengths = [len(t.split()) for (_, t, _) in self._events]

        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)
        conf_trend = confidences[-1] - confidences[0] if len(confidences) > 1 else 0.0
        avg_len = sum(lengths) / len(lengths)

        return {
            "avg_confidence": avg_conf,
            "min_confidence": min_conf,
            "max_confidence": max_conf,
            "confidence_trend": conf_trend,
            "avg_length": avg_len,
        }