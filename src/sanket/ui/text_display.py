"""Accessible real-time text output, subtitles, and transcript buffer (SANKET-047).

Provides:
1. Dual-language gloss translation (English and Hindi Devanagari) for the 16 core ISL signs.
2. Formatted subtitle stream with configurable case formatting and natural spacing.
3. Transcript history buffer with timestamps, search, clear, and rolling window eviction.
4. Duplicate emission suppression (hysteresis) to prevent a held sign from repeating.
5. Structured event callbacks for decoupled terminal or GUI display integration.
"""

from __future__ import annotations

import collections
import datetime
import enum
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sanket.realtime.engine import InferenceResult
from sanket.realtime.smoothing import SmoothedPrediction
from sanket.realtime.thresholding import (
    IDLE_GLOSS,
    NO_SIGN_GLOSS,
    CalibratedPrediction,
)

logger = logging.getLogger(__name__)


class TextCase(str, enum.Enum):
    """Text casing choices for subtitle and transcript display."""

    UPPERCASE = "uppercase"
    LOWERCASE = "lowercase"
    TITLECASE = "titlecase"
    SENTENCE = "sentence"


@dataclass(frozen=True)
class GlossTranslation:
    """Bilingual vocabulary entry for an Indian Sign Language gloss."""

    gloss_id: str
    english_label: str
    hindi_label: str
    hindi_transliteration: str
    description: str


# Canonical 16-sign ISL bilingual dictionary from docs/research/INITIAL_VOCABULARY.md
ISL_VOCABULARY_TRANSLATIONS: dict[str, GlossTranslation] = {
    "hello": GlossTranslation("hello", "Hello", "नमस्ते", "Namaste", "Greetings"),
    "namaste": GlossTranslation("namaste", "Hello", "नमस्ते", "Namaste", "Greetings"),
    "thank_you": GlossTranslation("thank_you", "Thank You", "धन्यवाद", "Dhanyavad", "Courtesy"),
    "please": GlossTranslation("please", "Please", "कृपया", "Kripya", "Courtesy"),
    "yes": GlossTranslation("yes", "Yes", "हाँ", "Haan", "Affirmation"),
    "no": GlossTranslation("no", "No", "नहीं", "Nahi", "Negation"),
    "help": GlossTranslation("help", "Help", "मदद", "Madad", "Emergency request"),
    "water": GlossTranslation("water", "Water", "पानी", "Paani", "Daily need"),
    "food": GlossTranslation("food", "Food", "खाना", "Khaana", "Daily need / eat"),
    "eat": GlossTranslation("eat", "Food", "खाना", "Khaana", "Daily need / eat"),
    "doctor": GlossTranslation("doctor", "Doctor", "चिकित्सक", "Doctor", "Healthcare"),
    "time": GlossTranslation("time", "Time", "समय", "Samay", "Temporal query"),
    "where": GlossTranslation("where", "Where", "कहाँ", "Kahan", "Location query"),
    "stop": GlossTranslation("stop", "Stop", "रुकिए", "Rukiye", "Directive"),
    "name": GlossTranslation("name", "Name", "नाम", "Naam", "Identity query"),
    "good": GlossTranslation("good", "Good", "अच्छा", "Achha", "Evaluation"),
    "bad": GlossTranslation("bad", "Bad", "बुरा", "Bura", "Evaluation"),
    "me": GlossTranslation("me", "Me", "मैं", "Main", "Pronoun / self"),
    "i": GlossTranslation("i", "Me", "मैं", "Main", "Pronoun / self"),
}


@dataclass(frozen=True)
class TranscriptEntry:
    """An individual recognized sign event stored in the transcript buffer."""

    gloss: str
    english_text: str
    hindi_text: str
    confidence: float
    timestamp_ms: float
    iso_timestamp: str
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextDisplayConfig:
    """Configuration options for accessible text transcript generation."""

    text_case: TextCase = TextCase.SENTENCE
    max_history_entries: int = 100
    suppress_duplicates: bool = True
    min_pause_between_emissions_ms: float = 350.0
    include_hindi: bool = True
    include_confidence: bool = False
    timestamp_format: str = "%H:%M:%S"


class AccessibleTextDisplay:
    """Accessible transcript engine and dual-language subtitle emitter.

    Collects real-time sign predictions, eliminates flutter and held-sign duplicate emissions,
    maps signs to English and Hindi, and dispatches subtitle events.
    """

    def __init__(self, config: TextDisplayConfig | None = None) -> None:
        """Initialize AccessibleTextDisplay with configuration.

        Args:
            config: Optional TextDisplayConfig instance.
        """
        self.config = config or TextDisplayConfig()
        self._history: collections.deque[TranscriptEntry] = collections.deque(
            maxlen=self.config.max_history_entries
        )
        self._subscribers: list[Callable[[TranscriptEntry], None]] = []

        # Emission suppression state
        self._last_emitted_gloss: str | None = None
        self._last_emission_timestamp_ms: float = 0.0
        self._current_hold_start_ms: float = 0.0

    @property
    def history(self) -> list[TranscriptEntry]:
        """Return a copy of the recorded transcript history."""
        return list(self._history)

    def subscribe(self, callback: Callable[[TranscriptEntry], None]) -> None:
        """Register an event listener invoked on every new sign emission.

        Args:
            callback: Callable accepting a `TranscriptEntry`.
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[TranscriptEntry], None]) -> None:
        """Remove a previously registered event listener."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def translate_gloss(self, gloss: str) -> tuple[str, str]:
        """Lookup bilingual translations for an ISL gloss.

        Args:
            gloss: Recognized class name or gloss identifier.

        Returns:
            Tuple of (english_translation, hindi_translation).
        """
        key = gloss.strip().lower()
        entry = ISL_VOCABULARY_TRANSLATIONS.get(key)
        if entry is not None:
            return entry.english_label, entry.hindi_label

        # Fallback for unrecognized or custom sign gloss
        cleaned = gloss.replace("_", " ").title()
        return cleaned, cleaned

    def format_text(self, text: str, case: TextCase | None = None) -> str:
        """Format an English string according to the requested text casing.

        Args:
            text: Text to format.
            case: Optional override casing mode.

        Returns:
            Formatted string.
        """
        mode = case or self.config.text_case
        if mode == TextCase.UPPERCASE:
            return text.upper()
        if mode == TextCase.LOWERCASE:
            return text.lower()
        if mode == TextCase.TITLECASE:
            return text.title()
        if mode == TextCase.SENTENCE:
            stripped = text.strip()
            if not stripped:
                return ""
            return stripped[0].upper() + stripped[1:].lower()
        return text

    def update(
        self,
        prediction: InferenceResult | SmoothedPrediction | CalibratedPrediction | str | None,
        confidence: float | None = None,
        timestamp_ms: float | None = None,
    ) -> TranscriptEntry | None:
        """Ingest a new prediction frame and conditionally emit a transcript entry.

        Applies duplicate suppression so a held sign is only emitted once until released.

        Args:
            prediction: Recognized sign instance or gloss string.
            confidence: Optional confidence score if string prediction provided.
            timestamp_ms: Optional timestamp in milliseconds.

        Returns:
            New `TranscriptEntry` if a valid non-duplicate sign was emitted, else None.
        """
        if prediction is None:
            return None

        # Extract normalized gloss and confidence
        raw_gloss, conf, ts = self._extract_prediction_info(prediction, confidence, timestamp_ms)

        # Ignore non-sign and idle states
        if raw_gloss.upper() in (NO_SIGN_GLOSS, IDLE_GLOSS, "NONE", ""):
            # Releasing hand resets last emitted sign after pause
            if self._last_emitted_gloss is not None:
                self._last_emitted_gloss = None
            return None

        # Duplicate suppression: do not re-emit the same sign if currently held
        if self.config.suppress_duplicates:
            if (
                self._last_emitted_gloss is not None
                and raw_gloss.lower() == self._last_emitted_gloss.lower()
            ):
                return None

            # Rate-limiting check between emissions
            time_since_last = ts - self._last_emission_timestamp_ms
            if time_since_last < self.config.min_pause_between_emissions_ms:
                return None

        # Translate gloss
        english_raw, hindi_raw = self.translate_gloss(raw_gloss)
        english_formatted = self.format_text(english_raw)

        # Build entry
        iso_str = datetime.datetime.now(datetime.timezone.utc).strftime(
            self.config.timestamp_format
        )
        entry = TranscriptEntry(
            gloss=raw_gloss.lower(),
            english_text=english_formatted,
            hindi_text=hindi_raw,
            confidence=conf,
            timestamp_ms=ts,
            iso_timestamp=iso_str,
        )

        # Record and update state
        self._history.append(entry)
        self._last_emitted_gloss = raw_gloss.lower()
        self._last_emission_timestamp_ms = ts

        # Dispatch to subscribers
        for callback in self._subscribers:
            try:
                callback(entry)
            except Exception as e:
                logger.error("Error in AccessibleTextDisplay subscriber callback: %s", e)

        return entry

    def get_full_transcript(
        self,
        separator: str = " ",
        language: str = "english",
    ) -> str:
        """Compose all entries in history into a single connected string.

        Args:
            separator: Join separator between words.
            language: 'english' or 'hindi'.

        Returns:
            Concatenated transcript string.
        """
        if language.lower() == "hindi":
            words = [e.hindi_text for e in self._history]
        else:
            words = [e.english_text for e in self._history]
        return separator.join(words)

    def clear(self) -> None:
        """Clear recorded transcript history and reset emission state."""
        self._history.clear()
        self._last_emitted_gloss = None
        self._last_emission_timestamp_ms = 0.0

    def _extract_prediction_info(
        self,
        prediction: InferenceResult | SmoothedPrediction | CalibratedPrediction | str,
        confidence: float | None,
        timestamp_ms: float | None,
    ) -> tuple[str, float, float]:
        """Extract uniform (gloss, confidence, timestamp_ms) from prediction types."""
        if isinstance(prediction, CalibratedPrediction):
            # If rejected, respect the gated rejection label (NO_SIGN or IDLE)
            gloss = prediction.gloss if not prediction.is_valid_sign else prediction.raw_gloss
            conf = prediction.confidence
            ts = timestamp_ms or prediction.timestamp_ms
        elif isinstance(prediction, SmoothedPrediction):
            gloss = prediction.active_gloss
            conf = prediction.confidence
            ts = timestamp_ms or prediction.timestamp_ms
        elif isinstance(prediction, InferenceResult):
            gloss = prediction.predicted_gloss
            conf = prediction.confidence
            ts = timestamp_ms or prediction.timestamp_ms
        elif isinstance(prediction, str):
            gloss = prediction
            conf = confidence if confidence is not None else 1.0
            ts = timestamp_ms if timestamp_ms is not None else 0.0
        else:
            raise TypeError(f"Unsupported prediction type: {type(prediction)}")

        return gloss, float(conf), float(ts)
