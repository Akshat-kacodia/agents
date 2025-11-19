from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

@dataclass
class InterruptionEvent:
    timestamp: datetime
    transcript: str
    confidence: float
    was_agent_speaking: bool
    decision: str  # "IGNORE" | "INTERRUPT" | "REGISTER"
    reason: str
    language: Optional[str] = "en"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "transcript": self.transcript,
            "confidence": self.confidence,
            "was_agent_speaking": self.was_agent_speaking,
            "decision": self.decision,
            "reason": self.reason,
            "language": self.language,
        }