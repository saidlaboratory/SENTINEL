"""Layer 4: LM Studio OpenAI-compatible interface."""

from __future__ import annotations

import logging
from typing import Any

from openai import APITimeoutError, OpenAI, OpenAIError

LOGGER = logging.getLogger(__name__)


class LocalLLM:
    """OpenAI SDK client for LM Studio local server."""

    def __init__(self, config: dict[str, Any]) -> None:
        lm_config = config["lm_studio"]
        self.model: str = str(lm_config["model"])
        self.temperature: float = float(lm_config["temperature"])
        self.max_tokens: int = int(lm_config["max_tokens"])
        self.client = OpenAI(
            base_url=str(lm_config["base_url"]),
            api_key=str(lm_config["api_key"]),
            timeout=float(lm_config["timeout_seconds"]),
        )

    def generate(self, prompt: str) -> str:
        """Generate a response from the currently loaded LM Studio model."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Refuse harmful, "
                            "illegal, exploitative, or privacy-invasive requests."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            LOGGER.info("Layer 4 generation completed")
            return response.choices[0].message.content or ""
        except APITimeoutError:
            LOGGER.error("LM Studio request timed out.")
            return "The local model timed out before producing a response."
        except OpenAIError:
            LOGGER.error("LM Studio request failed.")
            return "The local model request failed. Check that LM Studio is running."
