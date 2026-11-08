"""Unit tests for accessible text display and dual-language transcript buffer (SANKET-047)."""

from __future__ import annotations

from sanket.realtime.engine import InferenceResult
from sanket.realtime.thresholding import (
    IDLE_CLASS_ID,
    IDLE_GLOSS,
    NO_SIGN_CLASS_ID,
    NO_SIGN_GLOSS,
    CalibratedPrediction,
)
from sanket.ui.text_display import (
    AccessibleTextDisplay,
    TextCase,
    TextDisplayConfig,
    TranscriptEntry,
)


def _make_dummy_inference(gloss: str = "namaste", ts_ms: float = 1000.0) -> InferenceResult:
    return InferenceResult(
        predicted_gloss=gloss,
        predicted_class=1,
        confidence=0.92,
        probabilities={gloss: 0.92},
        feature_latency_ms=1.0,
        model_latency_ms=2.0,
        total_latency_ms=3.0,
        timestamp_ms=ts_ms,
    )


class TestAccessibleTextDisplay:
    """Test suite for subtitle emission, bilingual mapping, and duplicate suppression."""

    def test_vocabulary_translation_coverage(self) -> None:
        display = AccessibleTextDisplay()
        # Verify 16 core ISL signs exist and map to Hindi and English
        expected_signs = [
            "hello",
            "thank_you",
            "please",
            "yes",
            "no",
            "help",
            "water",
            "food",
            "doctor",
            "time",
            "where",
            "stop",
            "name",
            "good",
            "bad",
            "me",
        ]
        for sign in expected_signs:
            eng, hin = display.translate_gloss(sign)
            assert eng is not None and len(eng) > 0
            assert hin is not None and len(hin) > 0

    def test_casing_options(self) -> None:
        display = AccessibleTextDisplay()
        assert display.format_text("hello world", TextCase.UPPERCASE) == "HELLO WORLD"
        assert display.format_text("HELLO WORLD", TextCase.LOWERCASE) == "hello world"
        assert display.format_text("hello world", TextCase.TITLECASE) == "Hello World"
        assert display.format_text("hello world", TextCase.SENTENCE) == "Hello world"

    def test_single_sign_emission_and_history(self) -> None:
        display = AccessibleTextDisplay()
        pred = _make_dummy_inference("namaste", ts_ms=1000.0)

        entry = display.update(pred)
        assert isinstance(entry, TranscriptEntry)
        assert entry.gloss == "namaste"
        assert entry.english_text == "Hello"
        assert entry.hindi_text == "नमस्ते"
        assert len(display.history) == 1

    def test_held_sign_duplicate_suppression(self) -> None:
        display = AccessibleTextDisplay(
            TextDisplayConfig(
                suppress_duplicates=True,
                min_pause_between_emissions_ms=300.0,
            )
        )

        # First frame emits
        e1 = display.update(_make_dummy_inference("namaste", ts_ms=1000.0))
        assert e1 is not None

        # Subsequent consecutive frames of same held sign suppressed
        e2 = display.update(_make_dummy_inference("namaste", ts_ms=1033.0))
        e3 = display.update(_make_dummy_inference("namaste", ts_ms=1066.0))
        assert e2 is None
        assert e3 is None
        assert len(display.history) == 1

        # Hand releases to IDLE
        idle_calibrated = CalibratedPrediction(
            gloss=IDLE_GLOSS,
            class_id=IDLE_CLASS_ID,
            confidence=0.8,
            entropy=0.2,
            kinetic_energy=0.001,
            is_valid_sign=False,
            is_resting=True,
            rejection_reason="kinetic_energy_below_threshold",
            raw_gloss="namaste",
            raw_confidence=0.8,
            probabilities={"namaste": 0.8},
            timestamp_ms=1400.0,
        )
        display.update(idle_calibrated)

        # Signing again after release should emit
        e4 = display.update(_make_dummy_inference("namaste", ts_ms=1800.0))
        assert e4 is not None
        assert len(display.history) == 2

    def test_rejection_no_sign_does_not_emit(self) -> None:
        display = AccessibleTextDisplay()
        calibrated = CalibratedPrediction(
            gloss=NO_SIGN_GLOSS,
            class_id=NO_SIGN_CLASS_ID,
            confidence=0.4,
            entropy=2.8,
            kinetic_energy=0.02,
            is_valid_sign=False,
            is_resting=False,
            rejection_reason="entropy_exceeds_threshold",
            raw_gloss="random",
            raw_confidence=0.4,
            probabilities={"random": 0.4},
            timestamp_ms=1000.0,
        )
        entry = display.update(calibrated)
        assert entry is None
        assert len(display.history) == 0

    def test_event_subscriber_dispatch(self) -> None:
        display = AccessibleTextDisplay()
        emitted_entries = []

        def on_sign(entry: TranscriptEntry) -> None:
            emitted_entries.append(entry)

        display.subscribe(on_sign)

        display.update(_make_dummy_inference("water", ts_ms=1000.0))
        display.update(_make_dummy_inference("help", ts_ms=1500.0))

        assert len(emitted_entries) == 2
        assert emitted_entries[0].gloss == "water"
        assert emitted_entries[1].gloss == "help"

        display.unsubscribe(on_sign)
        display.update(_make_dummy_inference("doctor", ts_ms=2000.0))
        # Subscriber should not receive third emission
        assert len(emitted_entries) == 2

    def test_full_transcript_composition(self) -> None:
        display = AccessibleTextDisplay()
        display.update(_make_dummy_inference("hello", ts_ms=1000.0))
        display.update(_make_dummy_inference("water", ts_ms=1500.0))
        display.update(_make_dummy_inference("please", ts_ms=2000.0))

        eng_text = display.get_full_transcript(separator=" ", language="english")
        hin_text = display.get_full_transcript(separator=" ", language="hindi")

        assert eng_text == "Hello Water Please"
        assert hin_text == "नमस्ते पानी कृपया"

    def test_clear_buffer(self) -> None:
        display = AccessibleTextDisplay()
        display.update(_make_dummy_inference("hello", ts_ms=1000.0))
        assert len(display.history) == 1
        display.clear()
        assert len(display.history) == 0
        assert display.get_full_transcript() == ""
