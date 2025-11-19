

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from event import InterruptionEvent
logger = logging.getLogger(__name__)

class FillerClassifier:
    """
    Lightweight multi-language filler recognition.

    It does two things:
      1. Exact filler lookup (e.g., "uh", "umm", "haan").
      2. Simple fuzzy detection for stretched sounds like "ummmm", "hmmmmmm".
    """

    def __init__(self) -> None:
        # Per-language filler lexicon.
        # You can extend these at runtime via add_fillers.
        self._fillers_by_lang: Dict[str, Set[str]] = {
            "en": {
                "uh",
                "um",
                "umm",
                "hmm",
                "hm",
                "ah",
                "er",
                "erm",
                "like",
                "you know",
                "yeah",
                "ok",
            },
            "hi": {
                "haan",
                "haan ji",
                "acha",
                "theek",
                "arre",
                "toh",
            },
            "es": {"eh", "este", "pues", "bueno", "mm"},
            "fr": {"euh", "bah", "ben", "hein"},
        }

        # "Hard" interruption words. If they appear, we treat as intentional.
        self._priority_tokens: Set[str] = {
            # English
            "wait",
            "stop",
            "hold",
            "pause",
            "cancel",
            "hang",
            "no",
            # Hindi / Hinglish-ish
            "ruk",
            "ruko",
            "nahi",
            "bas",
            "thambo",  # some dialects
        }

        # Very rough patterns that suggest meaningful speech
        self._meaningful_patterns: List[re.Pattern[str]] = [
            re.compile(r"\b(what|how|when|where|why|who)\b", re.IGNORECASE),
            re.compile(r"\b(can|could|would|should|will|do|does|did)\b", re.IGNORECASE),
            re.compile(r"\b(please|sorry|excuse|thanks|thank)\b", re.IGNORECASE),
            re.compile(r"\b(yes|okay|ok|sure|alright)\b", re.IGNORECASE),
        ]

    # --- public API --------------------------------------------------------

    def add_fillers(self, language: str, words: List[str]) -> None:
        lang = language.lower()
        if lang not in self._fillers_by_lang:
            self._fillers_by_lang[lang] = set()
        new_words = {w.strip().lower() for w in words if w.strip()}
        self._fillers_by_lang[lang].update(new_words)
        logger.info("Added %d fillers for language=%s", len(new_words), lang)

    def all_fillers(self) -> Set[str]:
        out: Set[str] = set()
        for words in self._fillers_by_lang.values():
            out.update(words)
        return out

    def contains_priority_keyword(self, text: str) -> bool:
        tokens = self._tokenize(text)
        return any(tok in self._priority_tokens for tok in tokens)

    def looks_meaningful(self, text: str) -> bool:
        normalized = text.strip().lower()
        if not normalized:
            return False
        for pattern in self._meaningful_patterns:
            if pattern.search(normalized):
                return True
        return False

    def utterance_is_pure_filler(self, text: str, lang: str = "en") -> bool:
        """
        Returns True if the text is "only filler" (umm-like things / short dysfluencies).
        - For a single token: must be filler.
        - For multi-token: require that the majority are fillers.
        """
        normalized = text.strip().lower()
        if not normalized:
            return True

        tokens = self._tokenize(normalized)
        if not tokens:
            return True

        # If meaningful phrase, we call it NOT filler.
        if self.looks_meaningful(normalized):
            return False

        # Merge language-specific fillers with English as shared base.
        lang = lang.lower()
        lang_fillers = self._fillers_by_lang.get(lang, set())
        all_fillers = lang_fillers | self._fillers_by_lang.get("en", set())

        filler_count = 0
        for token in tokens:
            if self._is_token_filler(token, all_fillers):
                filler_count += 1

        if len(tokens) == 1:
            return filler_count == 1

        ratio = filler_count / len(tokens)
        # For multi-token, be generous: 80% of tokens must look like filler.
        return ratio >= 0.8

    # --- helpers -----------------------------------------------------------

    def _tokenize(self, txt: str) -> List[str]:
        return re.findall(r"\b\w+\b", txt.lower())

    def _is_token_filler(self, token: str, fillers: Set[str]) -> bool:
        # Exact lexicon match
        if token in fillers:
            return True

        # Approximate check: repeated characters, e.g. "ummmm", "hmmmmmm"
        squashed = self._squash_repetitions(token)
        if squashed in fillers:
            return True

        # "token" might include some repeated letters of a filler base form as prefix, e.g. "haaan"
        for candidate in fillers:
            if len(candidate) >= 2 and squashed.startswith(candidate[:2]):
                # token is built mainly from characters in "candidate"
                if len(set(token) - set(candidate)) == 0:
                    return True

        return False

    def _squash_repetitions(self, token: str) -> str:
        """
        Reduces runs of the same character:
          "ummmm" -> "um"
          "haaan" -> "han" (still close to "haan")
        """
        out_chars: List[str] = []
        last: Optional[str] = None
        for ch in token:
            if ch != last:
                out_chars.append(ch)
                last = ch
        return "".join(out_chars)