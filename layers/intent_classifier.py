"""Layer 1: semantic intent classification."""

from __future__ import annotations

import logging
from typing import Any

from transformers.pipelines import Pipeline

from layers.model_manager import ModelManager

LOGGER = logging.getLogger(__name__)


class IntentClassifier:
    """Zero-shot classifier for prompt intent."""

    LABELS: list[str] = ["benign", "borderline", "adversarial", "harmful"]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.classifier: Pipeline = ModelManager.zero_shot_pipeline()
        thresholds = config["thresholds"]
        self.block_labels: set[str] = set(thresholds["intent_block_labels"])
        self.block_confidence: float = float(
            thresholds["intent_block_confidence"]
        )

    def classify(self, prompt: str) -> dict[str, float | str | bool]:
        """Classify prompt intent."""
        try:
            result = self.classifier(prompt, self.LABELS, multi_label=False)
            label = str(result["labels"][0])
            confidence = float(result["scores"][0])
            blocked = label in self.block_labels and confidence >= self.block_confidence
            LOGGER.info("Layer 1 classification completed")
            return {
                "label": label,
                "confidence": confidence,
                "blocked": blocked,
            }
        except Exception:
            LOGGER.exception("Layer 1 classification failed.")
            return {
                "label": "harmful",
                "confidence": 1.0,
                "blocked": True,
            }
