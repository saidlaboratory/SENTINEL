"""Layer 2: encoding and obfuscation normalization."""

from __future__ import annotations

import base64
import binascii
import html
import logging
import re
import unicodedata
from codecs import decode as codec_decode
from typing import Any
from urllib.parse import unquote

LOGGER = logging.getLogger(__name__)


HOMOGLYPHS: dict[str, str] = {
    "а": "a",
    "е": "e",
    "о": "o",
    "р": "p",
    "с": "c",
    "х": "x",
    "у": "y",
    "і": "i",
    "ј": "j",
    "ѕ": "s",
    "Α": "A",
    "Β": "B",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Ι": "I",
    "Κ": "K",
    "Μ": "M",
    "Ν": "N",
    "Ο": "O",
    "Ρ": "P",
    "Τ": "T",
    "Χ": "X",
}

LEETSPEAK: dict[str, str] = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "@": "a",
    "$": "s",
    "!": "i",
}

SUSPICIOUS_ROT13_TERMS: set[str] = {
    "ignore",
    "jailbreak",
    "dan",
    "developer",
    "malware",
    "exploit",
    "bypass",
    "harmful",
    "scam",
}

SAFETY_TERMS: set[str] = {
    "ammonia",
    "autism",
    "bleach",
    "botnet",
    "blueprint",
    "blueprints",
    "bypass",
    "cancer",
    "cdc",
    "credential",
    "credentials",
    "disinformation",
    "etonitazene",
    "exploit",
    "fabricated",
    "fentanyl",
    "fraud",
    "harmful",
    "hoax",
    "isotonitazene",
    "jailbreak",
    "keylogger",
    "layout",
    "malware",
    "methamphetamine",
    "misinformation",
    "nitazene",
    "nitazenes",
    "opioid",
    "opioids",
    "payload",
    "phishing",
    "floorplan",
    "floorplans",
    "permission",
    "relapse",
    "sober",
    "sobriety",
    "precursor",
    "precursors",
    "ransomware",
    "scam",
    "synthesis",
    "synthesize",
    "smishing",
    "stealer",
    "trojan",
    "unsafe",
    "vaccines",
    "vinegar",
    "vishing",
}


class Normalizer:
    """Normalize encoded, obfuscated, or visually deceptive prompts."""

    _base64_pattern = re.compile(
        r"(?<![A-Za-z0-9+/=])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/=])"
    )
    _word_pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9@$!]{3,}\b")

    def __init__(self, config: dict[str, Any]) -> None:
        self.max_rounds: int = int(config["normalization"]["max_rounds"])

    def normalize(self, prompt: str) -> dict[str, str | list[str]]:
        """Repeatedly normalize until fixed point or max rounds."""
        current = prompt
        transformations: list[str] = []
        rot13_used = False

        for _ in range(self.max_rounds):
            changed = False

            updated = unicodedata.normalize("NFKC", current)
            if updated != current:
                current = updated
                transformations.append("unicode_nfkc")
                changed = True

            updated = "".join(HOMOGLYPHS.get(char, char) for char in current)
            if updated != current:
                current = updated
                transformations.append("homoglyph")
                changed = True

            updated = html.unescape(current)
            if updated != current:
                current = updated
                transformations.append("html_entity")
                changed = True

            updated = unquote(current)
            if updated != current:
                current = updated
                transformations.append("url_decode")
                changed = True

            updated = self._decode_base64_fragments(current)
            if updated != current:
                current = updated
                transformations.append("base64_decode")
                changed = True

            updated = self._normalize_leetspeak(current)
            if updated != current:
                current = updated
                transformations.append("leetspeak")
                changed = True

            updated = self._normalize_safety_typos(current)
            if updated != current:
                current = updated
                transformations.append("fuzzy_safety_terms")
                changed = True

            if not rot13_used:
                updated = codec_decode(current, "rot_13")
                if self._rot13_is_useful(current, updated):
                    current = updated
                    transformations.append("rot13")
                    rot13_used = True
                    changed = True

            if not changed:
                break

        LOGGER.info("Layer 2 normalization completed")
        return {
            "original": prompt,
            "normalized": current,
            "transformations": transformations,
        }

    def _decode_base64_fragments(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            candidate = match.group(1)
            try:
                padding = "=" * (-len(candidate) % 4)
                decoded = base64.b64decode(
                    candidate + padding,
                    validate=True,
                )
                decoded_text = decoded.decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                return candidate
            if self._looks_like_text(decoded_text):
                return decoded_text
            return candidate

        return self._base64_pattern.sub(replace, text)

    @staticmethod
    def _looks_like_text(value: str) -> bool:
        if not value.strip():
            return False
        printable = sum(char.isprintable() or char.isspace() for char in value)
        return printable / max(len(value), 1) > 0.9

    @staticmethod
    def _normalize_leetspeak(text: str) -> str:
        words = re.split(r"(\W+)", text)
        normalized: list[str] = []
        for word in words:
            if any(char in LEETSPEAK for char in word):
                normalized.append("".join(LEETSPEAK.get(char, char) for char in word))
            else:
                normalized.append(word)
        return "".join(normalized)

    @classmethod
    def _normalize_safety_typos(cls, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            lowered = token.lower()
            if lowered in SAFETY_TERMS:
                return token

            best_term = ""
            best_distance = 3
            for term in SAFETY_TERMS:
                if abs(len(lowered) - len(term)) > 1:
                    continue
                if lowered[0] != term[0]:
                    continue
                distance = cls._bounded_levenshtein(lowered, term, 2)
                if distance < best_distance:
                    best_distance = distance
                    best_term = term

            if best_distance <= 1 or (
                best_distance == 2 and len(lowered) >= 8
            ):
                return cls._match_case(token, best_term)
            return token

        return cls._word_pattern.sub(replace, text)

    @staticmethod
    def _bounded_levenshtein(source: str, target: str, limit: int) -> int:
        if abs(len(source) - len(target)) > limit:
            return limit + 1

        previous = list(range(len(target) + 1))
        for source_index, source_char in enumerate(source, start=1):
            current = [source_index]
            row_min = current[0]
            for target_index, target_char in enumerate(target, start=1):
                substitution_cost = 0 if source_char == target_char else 1
                current_value = min(
                    current[target_index - 1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + substitution_cost,
                )
                current.append(current_value)
                row_min = min(row_min, current_value)
            if row_min > limit:
                return limit + 1
            previous = current
        return previous[-1]

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.title()
        return replacement

    @staticmethod
    def _rot13_is_useful(original: str, transformed: str) -> bool:
        original_terms = set(re.findall(r"[a-z]+", original.lower()))
        transformed_terms = set(re.findall(r"[a-z]+", transformed.lower()))
        return bool(SUSPICIOUS_ROT13_TERMS & transformed_terms) and not bool(
            SUSPICIOUS_ROT13_TERMS & original_terms
        )
