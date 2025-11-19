"""
Interrupt handler for LiveKit voice agents.

Responsibilities:
- Classify user utterances as:
    * "IGNORE"      -> drop (filler or noise)
    * "INTERRUPT"   -> immediately stop the agent
    * "REGISTER"    -> treat as valid speech
- Take into account:
    * agent speaking state
    * confidence score from STT
    * filler vs meaningful text
    * multi-language fillers (English, Hindi, etc.)
    * optional small context window for statistics
"""

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
from context import ContextWindow

logger = logging.getLogger(__name__)

class InterruptHandler:
    """
    Core decision engine.

    High-level rules:

      1. If confidence < threshold          -> IGNORE ("noise")
      2. If priority keyword in text        -> INTERRUPT
      3. If agent is speaking:
         - pure filler                      -> IGNORE
         - otherwise                        -> INTERRUPT
      4. If agent is NOT speaking:
         - we REGISTER everything (filler or not),
           since that is just user speaking in their turn.
    """

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.6,
        enable_contextual_analysis: bool = True,
        enable_multi_language: bool = True,
        log_events: bool = True,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.enable_contextual_analysis = bool(enable_contextual_analysis)
        self.enable_multi_language = bool(enable_multi_language)
        self.log_events = bool(log_events)

        self._filler = FillerClassifier()
        self._context = ContextWindow() if enable_contextual_analysis else None

        self._agent_speaking: bool = False

        # stats
        self._events: List[InterruptionEvent] = []
        self._stats: Dict[str, float] = {
            "total_events": 0,
            "ignored_fillers": 0,
            "valid_interrupts": 0,
            "registered_speech": 0,
            "low_confidence_ignored": 0,
            "avg_confidence": 0.0,
            "fuzzy_estimated": 0.0,  # not strictly measured; just a metric bucket
        }

        logger.info(
            "InterruptHandler ready: threshold=%.2f, context=%s, multi_lang=%s",
            self.confidence_threshold,
            self.enable_contextual_analysis,
            self.enable_multi_language,
        )

    # --- configuration -----------------------------------------------------

    def set_agent_speaking(self, speaking: bool) -> None:
        """Update whether TTS is currently playing."""
        self._agent_speaking = bool(speaking)
        logger.debug("Agent speaking changed to %s", self._agent_speaking)

    def add_custom_fillers(self, language: str, fillers: List[str]) -> None:
        """Extend filler lexicon at runtime."""
        self._filler.add_fillers(language, fillers)

    # --- decision logic ----------------------------------------------------

    async def process_speech_event(
        self,
        transcript: str,
        confidence: float,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Main entrypoint used by your voice agent.

        Returns a dict:
            {
              "action": "IGNORE" | "INTERRUPT" | "REGISTER",
              "reason": "...",
              "should_stop_agent": bool,
              "metadata": {...}
            }
        """
        ts = datetime.now()
        language = (language or "en").lower()
        confidence = float(confidence)

        self._stats["total_events"] += 1

        # Update rolling average confidence
        t = self._stats["total_events"]
        prev_avg = self._stats["avg_confidence"]
        self._stats["avg_confidence"] = ((prev_avg * (t - 1)) + confidence) / t

        if self._context is not None:
            self._context.add(transcript, confidence, ts)
            context_snapshot = self._context.snapshot()
        else:
            context_snapshot = {}

        decision, reason = self._decide(transcript, confidence, language)

        event = InterruptionEvent(
            timestamp=ts,
            transcript=transcript,
            confidence=confidence,
            was_agent_speaking=self._agent_speaking,
            decision=decision,
            reason=reason,
            language=language,
        )
        if self.log_events:
            self._events.append(event)
            logger.info(
                "[%s] '%s' (conf=%.2f, lang=%s, agent_speaking=%s) -> %s",
                decision,
                transcript,
                confidence,
                language,
                self._agent_speaking,
                reason,
            )

        if decision == "IGNORE":
            if "confidence" in reason or "noise" in reason:
                self._stats["low_confidence_ignored"] += 1
            else:
                self._stats["ignored_fillers"] += 1
        elif decision == "INTERRUPT":
            self._stats["valid_interrupts"] += 1
        elif decision == "REGISTER":
            self._stats["registered_speech"] += 1

        return {
            "action": decision,
            "reason": reason,
            "should_stop_agent": decision == "INTERRUPT",
            "metadata": {
                "confidence": confidence,
                "language": language,
                "was_agent_speaking": self._agent_speaking,
                "context": context_snapshot,
            },
        }

    def _decide(self, transcript: str, confidence: float, language: str) -> Tuple[str, str]:
        text = (transcript or "").strip()

        # 1) obvious background noise / random artifacts
        if confidence < self.confidence_threshold:
            return (
                "IGNORE",
                f"low_confidence({confidence:.2f} < {self.confidence_threshold:.2f})",
            )

        # 2) explicit interruption cues ("stop", "wait", "ruko")
        if self._filler.contains_priority_keyword(text):
            # If the agent is quiet, it's still meaningful user speech,
            # but decision remains INTERRUPT so that code can choose to stop TTS if running.
            return "INTERRUPT", "priority_command"

        # 3) filler vs content
        is_pure_filler = self._filler.utterance_is_pure_filler(text, language)

        # 4) when agent is speaking
        if self._agent_speaking:
            if is_pure_filler:
                return "IGNORE", "filler_while_agent_speaking"
            return "INTERRUPT", "meaningful_speech_while_agent_speaking"

        # 5) when agent is not speaking
        if not self._agent_speaking:
            if is_pure_filler:
                return "REGISTER", "filler_while_agent_silent"
            return "REGISTER", "speech_while_agent_silent"

        # fallback (should not hit)
        return "REGISTER", "default"

    # --- statistics / export -----------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        total = max(1, int(self._stats["total_events"]))
        ignored = self._stats["ignored_fillers"]
        interrupts = self._stats["valid_interrupts"]

        return {
            **self._stats,
            "ignore_rate": ignored / total,
            "interrupt_rate": interrupts / total,
        }

    def export_event_log(self, path: str = "interruption_log.json") -> None:
        data = [ev.to_dict() for ev in self._events]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Exported %d events to %s", len(self._events), path)

    def print_summary(self) -> None:
        stats = self.get_statistics()
        print("\n" + "-" * 60)
        print("INTERRUPT HANDLER SUMMARY")
        print("-" * 60)
        print(f"Total events       : {int(stats['total_events'])}")
        print(f"Ignored fillers    : {int(stats['ignored_fillers'])}")
        print(f"Valid interrupts   : {int(stats['valid_interrupts'])}")
        print(f"Registered speech  : {int(stats['registered_speech'])}")
        print(f"Low conf ignored   : {int(stats['low_confidence_ignored'])}")
        print(f"Avg confidence     : {stats['avg_confidence']:.2f}")
        print(f"Ignore rate        : {stats['ignore_rate']:.1%}")
        print(f"Interrupt rate     : {stats['interrupt_rate']:.1%}")
        print("-" * 60 + "\n")


# Quick manual check hook
if __name__ == "__main__":
    import asyncio

    async def quick_demo() -> None:
        h = InterruptHandler()
        h.set_agent_speaking(True)
        tests = [
            ("ummmm", 0.9, "en"),
            ("wait one second", 0.9, "en"),
            ("haan", 0.88, "hi"),
            ("how does that work", 0.95, "en"),
        ]
        for txt, conf, lang in tests:
            r = await h.process_speech_event(txt, conf, lang)
            print(f"{txt!r} -> {r['action']} ({r['reason']})")
        h.print_summary()

    asyncio.run(quick_demo())
