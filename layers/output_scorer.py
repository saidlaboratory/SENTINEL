"""Layer 5: output harm scoring."""

from __future__ import annotations

import logging
from typing import Any

from transformers.pipelines import Pipeline

from layers.model_manager import ModelManager

LOGGER = logging.getLogger(__name__)


class OutputScorer:
    """Zero-shot scorer for generated text."""

    LABELS: list[str] = ["safe", "borderline", "harmful"]

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.classifier: Pipeline = ModelManager.zero_shot_pipeline()
        self.harmful_threshold: float = float(
            config["thresholds"]["output_harmful_confidence"]
        )
        self.refusal_message: str = str(config["app"]["refusal_message"])

    def score(self, output: str) -> dict[str, float | str | bool]:
        """Score model output and decide whether to replace it."""
        try:
            result = self.classifier(output, self.LABELS, multi_label=False)
            label = str(result["labels"][0])
            confidence = float(result["scores"][0])
            blocked = label == "harmful" and confidence >= self.harmful_threshold
            LOGGER.info("Layer 5 scoring completed")
            return {
                "label": label,
                "confidence": confidence,
                "blocked": blocked,
                "response": self.refusal_message if blocked else output,
            }
        except Exception:
            LOGGER.exception("Layer 5 output scoring failed.")
            return {
                "label": "harmful",
                "confidence": 1.0,
                "blocked": True,
                "response": self.refusal_message,
            }
