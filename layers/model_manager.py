"""Shared one-time model loading for IAJDA.

This module is the only place that should construct heavyweight ML models.
Layer classes receive cached objects from here, which prevents per-request
pipeline(), from_pretrained(), or SentenceTransformer() reloads.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines import Pipeline

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Runtime device metadata."""

    cuda_available: bool
    device_name: str
    pipeline_device: int


class ModelManager:
    """Singleton-style cache for all IAJDA ML models."""

    _lock: Lock = Lock()
    _device_info: DeviceInfo | None = None
    _nli_model: Any | None = None
    _nli_tokenizer: Any | None = None
    _zero_shot_pipeline: Pipeline | None = None
    _embedding_model: SentenceTransformer | None = None
    _loaded: bool = False

    @classmethod
    def device_info(cls) -> DeviceInfo:
        """Return cached CUDA/CPU selection."""
        if cls._device_info is None:
            cuda_available = torch.cuda.is_available()
            device_name = "cuda" if cuda_available else "cpu"
            cls._device_info = DeviceInfo(
                cuda_available=cuda_available,
                device_name=device_name,
                pipeline_device=0 if cuda_available else -1,
            )
        return cls._device_info

    @classmethod
    def initialize(
        cls,
        nli_model_name: str,
        embedding_model_name: str,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        """Load all models exactly once at process startup."""
        with cls._lock:
            if cls._loaded:
                return

            device = cls.device_info()
            LOGGER.info("CUDA available: %s", device.cuda_available)
            LOGGER.info("Using device: %s", device.device_name)
            if progress_callback is not None:
                progress_callback(f"CUDA available: {device.cuda_available}")
                progress_callback(f"Using device: {device.device_name}")

            LOGGER.info("[STARTUP] Loading Layer 1 model...")
            if progress_callback is not None:
                progress_callback("Loading Layer 1 model...")
            cls._nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            cls._nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_model_name
            )
            cls._nli_model.to(device.device_name)
            cls._nli_model.eval()
            cls._zero_shot_pipeline = pipeline(
                task="zero-shot-classification",
                model=cls._nli_model,
                tokenizer=cls._nli_tokenizer,
                device=device.pipeline_device,
            )

            LOGGER.info("[STARTUP] Loading Layer 3 model...")
            if progress_callback is not None:
                progress_callback("Loading Layer 3 model...")
            cls._embedding_model = SentenceTransformer(
                embedding_model_name,
                device=device.device_name,
            )

            LOGGER.info("[STARTUP] Loading Layer 5 model...")
            LOGGER.info(
                "[STARTUP] Reusing Layer 1 DeBERTa instance for Layer 5."
            )
            if progress_callback is not None:
                progress_callback("Loading Layer 5 model...")
                progress_callback("Reusing Layer 1 DeBERTa instance for Layer 5.")
            cls._loaded = True
            LOGGER.info("[STARTUP] Models loaded successfully.")
            if progress_callback is not None:
                progress_callback("Models loaded successfully.")

    @classmethod
    def zero_shot_pipeline(cls) -> Pipeline:
        """Return the shared zero-shot pipeline."""
        if cls._zero_shot_pipeline is None:
            raise RuntimeError("ModelManager.initialize() must run first.")
        return cls._zero_shot_pipeline

    @classmethod
    def embedding_model(cls) -> SentenceTransformer:
        """Return the shared sentence embedding model."""
        if cls._embedding_model is None:
            raise RuntimeError("ModelManager.initialize() must run first.")
        return cls._embedding_model
