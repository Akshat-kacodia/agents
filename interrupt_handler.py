"""
Voice Interruption Handler for LiveKit Agents
============================================

- Detects filler sounds ("umm", "hmmm", "haan", etc.) with fuzzy matching
- Handles multiple languages (en, hi, es, fr)
- Distinguishes between:
    * IGNORE      -> noise / filler while agent is speaking
    * INTERRUPT   -> real interruption (e.g., "wait", "stop", "ruko")
    * REGISTER    -> valid speech to be handled by the agent
"""

import re
import logging
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class InterruptionEvent:
    timestamp: datetime
    transcript: str
    confidence: float
    was_agent_speaking: bool
    decision: str  # "IGNORE", "INTERRUPT", "REGISTER"
    reason: str
    language: Optional[str] = "en"

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "transcript": self.transcript,
            "confidence": self.confidence,
            "was_agent_speaking": self.was_agent_speaking,
            "decision": self.decision,
            "reason": self.reason,
            "language": self.language,
        }


class MultiLanguageFillerDetector:
    """
    Handles any variation of filler sounds (umm, ummm, ummmm, etc.)
    and supports multiple languages.
    """

    def __init__(self) -> None:
        # Base filler vocabulary per language
        self.filler_words: Dict[str, Set[str]] = {
            "en": {"uh", "um", "umm", "hmm", "hm", "ah", "er", "erm", "like", "you know", "yeah"},
            "hi": {"haan", "hmm", "acha", "theek", "haan ji", "arrey", "toh"},
            "es": {"eh", "este", "pues", "bueno", "mm"},
            "fr": {"euh", "beh", "bon", "hein"},
        }

        # Priority interruption words (both English + Hindi-ish)
        self.priority_words: Set[str] = {
            "wait", "stop", "no", "hold", "pause", "cancel", "hang", "one", "second",
            "ruk", "ruko", "nahi", "bas",
        }

        # Patterns that indicate meaningful speech (questions, polite phrases, etc.)
        self.meaningful_patterns = [
            r"\b(what|how|when|where|why|who)\b",
            r"\b(can|could|would|should|will|do|does|did)\b",
            r"\b(please|sorry|excuse|thanks|thank)\b",
            r"\b(yes|okay|ok|sure|alright)\b",
        ]

    def add_filler_words(self, language: str, words: List[str]) -> None:
        """Dynamically add filler words for a language."""
        if language not in self.filler_words:
            self.filler_words[language] = set()
        self.filler_words[language].update(
            word.strip().lower() for word in words if word.strip()
        )
        logger.info("Added %d filler words for language '%s'", len(words), language)

    def is_filler_sound(self, word: str, fillers: Set[str]) -> bool:
        """
        Matches repetitions of core filler patterns, e.g.:

        - "um"  -> "um", "umm", "ummm", "ummmm"
        - "hm"  -> "hm", "hmm", "hmmm", "hmmmm"
        - "uh"  -> "uh", "uhh", "uhhh"
        - "haan" -> "haan", "haaan", "haaaan", ...
        """
        word = word.lower().strip()

        for filler in fillers:
            f_len = len(filler)

            # Very short fillers (1-2 chars): aggressive matching
            if f_len <= 2:
                filler_chars = set(filler)
                word_chars = set(word)
                if word_chars.issubset(filler_chars) and len(word) >= f_len:
                    return True

            # Medium fillers (length 3): e.g. "umm", "hmm"
            elif f_len == 3 and len(word) >= 2:
                first_char = filler[0]
                repeated_char = filler[1]
                if word[0] == first_char:
                    repeated_count = word.count(repeated_char)
                    if repeated_count >= 2 and repeated_count / len(word) >= 0.6:
                        return True

            # Longer fillers (>=4): check prefix and allowed chars
            else:
                core = filler[:2]
                if word.startswith(core):
                    filler_chars = set(filler)
                    word_chars = set(word)
                    if word_chars.issubset(filler_chars):
                        return True

        return False

    def is_filler_only(self, text: str, language: str = "en") -> bool:
        """
        True if the text is *only* filler / hesitation, False if it contains
        meaningful content.
        """
        if not text or not text.strip():
            return True

        normalized_text = text.lower().strip()
        words = normalized_text.split()

        # Priority commands are always meaningful
        if any(word in self.priority_words for word in words):
            return False

        # Check for meaningful patterns
        for pattern in self.meaningful_patterns:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                return False

        # Extract words
        tokens = re.findall(r"\b\w+\b", normalized_text)
        if not tokens:
            return True

        language_fillers = self.filler_words.get(language, set())
        english_fillers = self.filler_words.get("en", set())
        all_fillers = language_fillers | english_fillers

        filler_count = 0
        for token in tokens:
            if token in all_fillers or self.is_filler_sound(token, all_fillers):
                filler_count += 1

        threshold = 1.0 if len(tokens) == 1 else 0.8
        filler_ratio = filler_count / len(tokens)
        return filler_ratio >= threshold

    def contains_priority_command(self, text: str) -> bool:
        """Return True if text contains a 'hard' interruption like 'wait', 'stop', 'ruko'."""
        normalized = text.lower().strip()
        words = normalized.split()
        return any(word in self.priority_words for word in words)

    def get_all_fillers(self) -> Set[str]:
        all_fillers: Set[str] = set()
        for fillers in self.filler_words.values():
            all_fillers.update(fillers)
        return all_fillers


class ContextualBufferAnalyzer:
    """
    Keeps a short buffer of recent speech events and provides
    rough trends (confidence, complexity, etc.) for analysis.
    """

    def __init__(self, buffer_duration_ms: int = 150) -> None:
        self.buffer_duration_ms = buffer_duration_ms
        self.speech_buffer: deque = deque(maxlen=10)

    def add_event(self, transcript: str, confidence: float, timestamp: datetime) -> None:
        self.speech_buffer.append(
            {"transcript": transcript, "confidence": confidence, "timestamp": timestamp}
        )

    def get_context(self) -> Dict:
        if not self.speech_buffer:
            return {
                "avg_confidence": 0.0,
                "confidence_trend": 0.0,
                "word_count_trend": 0.0,
                "is_escalating": False,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            }

        events = list(self.speech_buffer)
        confidences = [e["confidence"] for e in events]
        word_counts = [len(e["transcript"].split()) for e in events]

        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        max_conf = max(confidences)

        confidence_trend = 0.0
        word_count_trend = 0.0
        if len(events) >= 2:
            confidence_trend = confidences[-1] - confidences[0]
            word_count_trend = word_counts[-1] - word_counts[0]

        is_escalating = confidence_trend > 0.1 and word_count_trend > 0

        return {
            "avg_confidence": avg_conf,
            "confidence_trend": confidence_trend,
            "word_count_trend": word_count_trend,
            "is_escalating": is_escalating,
            "min_confidence": min_conf,
            "max_confidence": max_conf,
        }


class InterruptHandler:
    """
    Core policy:

    1. Low confidence        -> IGNORE (noise)
    2. Priority command      -> INTERRUPT (always)
    3. Agent speaking+filler -> IGNORE
    4. Agent speaking+speech -> INTERRUPT
    5. Agent quiet           -> REGISTER (speech is valid)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        enable_contextual_analysis: bool = True,
        enable_multi_language: bool = True,
        log_events: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.enable_contextual_analysis = enable_contextual_analysis
        self.enable_multi_language = enable_multi_language
        self.log_events = log_events

        self.filler_detector = MultiLanguageFillerDetector()
        self.context_analyzer = (
            ContextualBufferAnalyzer() if enable_contextual_analysis else None
        )

        self.is_agent_speaking = False

        self.event_log: List[InterruptionEvent] = []
        self.stats = {
            "total_events": 0,
            "ignored_fillers": 0,
            "valid_interrupts": 0,
            "registered_speech": 0,
            "low_confidence_ignored": 0,
            "fuzzy_filler_matches": 0,
            "avg_confidence": 0.0,
        }

        logger.info("InterruptHandler initialized:")
        logger.info("  - Confidence threshold: %.2f", confidence_threshold)
        logger.info("  - Contextual analysis: %s", enable_contextual_analysis)
        logger.info("  - Multi-language: %s", enable_multi_language)
        logger.info("  - Aggressive fuzzy filler matching: ENABLED")

    def set_agent_speaking(self, speaking: bool) -> None:
        self.is_agent_speaking = speaking
        logger.debug("Agent speaking state: %s", speaking)

    def add_custom_fillers(self, language: str, fillers: List[str]) -> None:
        self.filler_detector.add_filler_words(language, fillers)

    async def process_speech_event(
        self,
        transcript: str,
        confidence: float,
        language: str = "en",
    ) -> Dict[str, object]:
        timestamp = datetime.now()
        self.stats["total_events"] += 1

        # Update average confidence
        prev_total = self.stats["total_events"] - 1
        total_conf = self.stats["avg_confidence"] * prev_total
        self.stats["avg_confidence"] = (total_conf + confidence) / max(
            self.stats["total_events"], 1
        )

        context = {}
        if self.context_analyzer:
            self.context_analyzer.add_event(transcript, confidence, timestamp)
            context = self.context_analyzer.get_context()

        decision, reason = await self._make_decision(
            transcript, confidence, language, context
        )

        ev = InterruptionEvent(
            timestamp=timestamp,
            transcript=transcript,
            confidence=confidence,
            was_agent_speaking=self.is_agent_speaking,
            decision=decision,
            reason=reason,
            language=language,
        )

        if self.log_events:
            self.event_log.append(ev)
            logger.info(
                "[%s] '%s' | conf=%.2f | %s", decision, transcript, confidence, reason
            )

        if decision == "IGNORE":
            self.stats["ignored_fillers"] += 1
            if "fuzzy" in reason.lower():
                self.stats["fuzzy_filler_matches"] += 1
        elif decision == "INTERRUPT":
            self.stats["valid_interrupts"] += 1
        elif decision == "REGISTER":
            self.stats["registered_speech"] += 1

        return {
            "action": decision,
            "reason": reason,
            "should_stop_agent": decision == "INTERRUPT",
            "metadata": {
                "confidence": confidence,
                "was_agent_speaking": self.is_agent_speaking,
                "language": language,
                "context": context,
            },
        }

    async def _make_decision(
        self,
        transcript: str,
        confidence: float,
        language: str,
        context: Dict,
    ) -> tuple[str, str]:
        # 1) Low confidence → noise
        if confidence < self.confidence_threshold:
            self.stats["low_confidence_ignored"] += 1
            return "IGNORE", f"low_confidence ({confidence:.2f} < {self.confidence_threshold})"

        # 2) Priority command (wait/stop/ruko/etc.) → always interrupt
        if self.filler_detector.contains_priority_command(transcript):
            return "INTERRUPT", "priority_command_detected"

        # 3) Filler-only text?
        is_filler_only = self.filler_detector.is_filler_only(transcript, language)

        # 4) Agent speaking + filler → IGNORE
        if self.is_agent_speaking and is_filler_only:
            return "IGNORE", "filler_while_agent_speaking (fuzzy match)"

        # 5) Agent speaking + meaningful speech → INTERRUPT
        if self.is_agent_speaking and not is_filler_only:
            return "INTERRUPT", "meaningful_speech_while_agent_speaking"

        # 6) Agent quiet → REGISTER
        if not self.is_agent_speaking:
            if is_filler_only:
                return "REGISTER", "filler_while_agent_quiet"
            return "REGISTER", "speech_while_agent_quiet"

        # Fallback
        return "REGISTER", "default_registration"

    def get_statistics(self) -> Dict:
        total = max(self.stats["total_events"], 1)
        ignored = self.stats["ignored_fillers"]
        interrupts = self.stats["valid_interrupts"]
        fuzzy = self.stats["fuzzy_filler_matches"]
        return {
            **self.stats,
            "ignore_rate": ignored / total,
            "interrupt_rate": interrupts / total,
            "fuzzy_match_rate": fuzzy / max(ignored, 1),
        }

    def export_event_log(self, filepath: str = "interruption_log.json") -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([ev.to_dict() for ev in self.event_log], f, indent=2)
        logger.info("Exported %d events to %s", len(self.event_log), filepath)

    def print_summary(self) -> None:
        stats = self.get_statistics()
        print("\n" + "-" * 60)
        print("INTERRUPT HANDLER - SUMMARY")
        print("-" * 60)
        print(f"Total events processed : {stats['total_events']}")
        print(
            f"Ignored fillers       : {stats['ignored_fillers']} "
            f"({stats['ignore_rate']:.1%})"
        )
        print(f"  └─ Fuzzy matches    : {stats['fuzzy_filler_matches']}")
        print(
            f"Valid interrupts      : {stats['valid_interrupts']} "
            f"({stats['interrupt_rate']:.1%})"
        )
        print(f"Registered speech     : {stats['registered_speech']}")
        print(f"Low confidence ignored: {stats['low_confidence_ignored']}")
        print(f"Average confidence    : {stats['avg_confidence']:.2f}")
        print("-" * 60 + "\n")


if __name__ == "__main__":
    # Tiny manual smoke test
    import asyncio

    async def _demo() -> None:
        h = InterruptHandler()
        h.set_agent_speaking(True)
        samples = [
            ("ummm", 0.9, "en"),
            ("wait one sec", 0.9, "en"),
            ("haan", 0.9, "hi"),
            ("how does this work", 0.95, "en"),
        ]
        for text, conf, lang in samples:
            res = await h.process_speech_event(text, conf, lang)
            print(text, "->", res["action"], res["reason"])

        h.print_summary()

    asyncio.run(_demo())
