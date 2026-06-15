"""IAJDA command-line application."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml

from layers.context_verifier import ContextVerifier
from layers.intent_classifier import IntentClassifier
from layers.llm_core import LocalLLM
from layers.model_manager import ModelManager
from layers.normalizer import Normalizer
from layers.output_scorer import OutputScorer

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
LOGGER = logging.getLogger(__name__)

SENTINEL_BANNER = [
    "=======================",
    "        SENTINEL       ",
    "=======================",
    "IAJDA Security Evaluation Prototype v1.8",
    "Ideal Anti-Jailbreak Defense Architecture",
]


class Ansi:
    """ANSI color helpers for PowerShell-friendly terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"


LAYER_LABELS: dict[str, str] = {
    "layer1_ms": "[Layer 1] Intent Classification",
    "layer2_ms": "[Layer 2] Input Normalization",
    "layer3_ms": "[Layer 3] Context Verification",
    "layer4_ms": "[Layer 4] LLM Generation",
    "layer5_ms": "[Layer 5] Output Validation",
}

MISINFORMATION_RULES: set[str] = {
    "hazardous_chemical_misinformation",
    "misinformation_generation",
    "deceptive_factual_edit",
    "evidence_distortion_edit",
    "defamatory_false_claim_generation",
    "generic_misinformation_persuasion",
    "generic_deceptive_rewrite",
    "generic_harmful_persuasion",
    "decoded_obfuscated_misinformation",
    "deceptive_persuasion_request",
    "public_health_misinformation",
    "demographic_harm_misinformation",
    "protected_class_persuasion_abuse",
}

CYBER_ABUSE_RULES: set[str] = {
    "direct_fraud_request",
    "direct_malware_request",
    "credential_theft_request",
}

CHEM_BIO_RULES: set[str] = {
    "controlled_substance_synthesis",
    "chemical_biological_weaponization",
    "actionable_chemistry_protocol",
}

ABUSE_EXPLOITATION_RULES: set[str] = {
    "addiction_relapse_encouragement",
    "recovery_vulnerability_abuse",
}

PHYSICAL_SECURITY_RULES: set[str] = {
    "unauthorized_facility_recon",
    "sensitive_site_intrusion_planning",
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def configure_logging(config: dict[str, Any], debug: bool = False) -> None:
    """Configure clean or debug logging."""
    level_name = str(config["logging"]["level"]).upper()
    level = getattr(logging, level_name, logging.INFO)
    root_level = level if debug else logging.ERROR
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=root_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    dependency_level = logging.INFO if debug else logging.ERROR
    for logger_name in (
        "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "urllib3",
        "openai",
    ):
        logging.getLogger(logger_name).setLevel(dependency_level)
    logging.getLogger("httpx").setLevel(logging.WARNING if debug else logging.ERROR)
    if not debug:
        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
        except Exception:
            pass
        try:
            from huggingface_hub.utils import logging as hub_logging

            hub_logging.set_verbosity_error()
        except Exception:
            pass


def color(text: str, ansi_code: str, enabled: bool = True) -> str:
    """Return ANSI-colored text when enabled."""
    if not enabled:
        return text
    return f"{ansi_code}{text}{Ansi.RESET}"


def print_banner(use_color: bool = True) -> None:
    """Print a centered academic-paper-friendly startup banner."""
    width = min(max(shutil.get_terminal_size((88, 24)).columns, 64), 100)
    print()
    for line in SENTINEL_BANNER:
        styled = color(line.center(width), Ansi.CYAN + Ansi.BOLD, use_color)
        print(styled)
    print()


def print_startup(message: str, use_color: bool = True) -> None:
    """Print compact startup progress."""
    prefix = color("[STARTUP]", Ansi.CYAN, use_color)
    print(f"{prefix} {message}")


def print_layer_status(
    layer_key: str,
    milliseconds: float,
    use_color: bool = True,
    skipped: bool = False,
) -> None:
    """Print one compact layer status line."""
    label = LAYER_LABELS[layer_key]
    if skipped:
        marker = color("-", Ansi.YELLOW, use_color)
        print(f"{label:<38} {marker} skipped")
        return
    check = color("✓", Ansi.GREEN, use_color)
    print(f"{label:<38} {check} {milliseconds / 1000:.2f}s")


def print_transformation_hint(
    transformations: list[str],
    use_color: bool = True,
) -> None:
    """Print compact Layer 2 transformation details in clean mode."""
    if not transformations:
        return
    label = color("Transformations", Ansi.YELLOW, use_color)
    print(f"{label:<38} : {', '.join(transformations)}")


class IAJDAPipeline:
    """Five-layer IAJDA safety wrapper."""

    def __init__(
        self,
        config: dict[str, Any],
        clean_mode: bool = True,
        use_color: bool = True,
    ) -> None:
        self.config = config
        self.clean_mode = clean_mode
        self.use_color = use_color
        performance_config = config.get("performance", {})
        self.fast_rule_gate = bool(performance_config.get("fast_rule_gate", True))
        self.skip_pre_llm_output_scoring = bool(
            performance_config.get(
                "skip_output_scoring_for_pre_llm_blocks",
                True,
            )
        )
        ModelManager.initialize(
            nli_model_name=str(config["models"]["nli_model"]),
            embedding_model_name=str(config["models"]["embedding_model"]),
            progress_callback=(
                lambda message: print_startup(message, use_color)
                if clean_mode
                else None
            ),
        )
        self.intent = IntentClassifier(config)
        self.normalizer = Normalizer(config)
        self.context = ContextVerifier(
            config,
            PROJECT_ROOT / "jailbreak_templates" / "templates.json",
        )
        self.llm = LocalLLM(config)
        self.output_scorer = OutputScorer(config)
        self.latency_csv = PROJECT_ROOT / str(config["app"]["latency_csv"])
        self.latency_csv.parent.mkdir(parents=True, exist_ok=True)

    def run(self, prompt: str) -> dict[str, Any]:
        """Run the five-layer pipeline with latency instrumentation."""
        total_start = time.perf_counter()
        timings: dict[str, float] = {}
        skipped_layers: set[str] = set()

        intent_result: dict[str, float | str | bool] = {
            "label": "skipped",
            "confidence": 0.0,
            "blocked": False,
        }
        output_result: dict[str, float | str | bool] = {
            "label": "skipped",
            "confidence": 0.0,
            "blocked": False,
            "response": str(self.config["app"]["refusal_message"]),
        }

        start = time.perf_counter()
        normalized = self.normalizer.normalize(prompt)
        timings["layer2_ms"] = self._elapsed_ms(start)

        normalized_prompt = str(normalized["normalized"])
        transformations = [
            str(item) for item in normalized.get("transformations", [])
        ]
        fast_context_result: dict[str, float | str | list[str] | bool] | None = None
        if self.fast_rule_gate:
            start = time.perf_counter()
            fast_context_result = self.context.verify_rules(
                normalized_prompt,
                transformations,
            )
            timings["layer3_ms"] = self._elapsed_ms(start)

        if fast_context_result is not None and bool(fast_context_result["blocked"]):
            timings["layer1_ms"] = 0.0
            timings["layer4_ms"] = 0.0
            timings["layer5_ms"] = 0.0
            skipped_layers.update({"layer1_ms", "layer4_ms", "layer5_ms"})
            context_result = fast_context_result
            raw_response = str(self.config["app"]["refusal_message"])
            output_result["response"] = raw_response
            pre_llm_blocked = True
            timings["total_ms"] = self._elapsed_ms(total_start)
            self._write_latency(prompt, timings)
            self._print_clean_layers(timings, skipped_layers, transformations)
            return {
                "final_response": raw_response,
                "intent": intent_result,
                "normalization": normalized,
                "context": context_result,
                "output": output_result,
                "blocked_before_llm": pre_llm_blocked,
                "latency": timings,
                "refusal_message": str(self.config["app"]["refusal_message"]),
            }

        start = time.perf_counter()
        intent_result = self.intent.classify(prompt)
        timings["layer1_ms"] = self._elapsed_ms(start)

        start = time.perf_counter()
        context_result = self.context.verify(normalized_prompt, transformations)
        timings["layer3_ms"] = self._elapsed_ms(start)

        pre_llm_blocked = bool(intent_result["blocked"]) or bool(
            context_result["blocked"]
        )
        if pre_llm_blocked:
            raw_response = str(self.config["app"]["refusal_message"])
            LOGGER.info("Layer 4 generation completed")
            timings["layer4_ms"] = 0.0
            skipped_layers.add("layer4_ms")
        else:
            start = time.perf_counter()
            raw_response = self.llm.generate(normalized_prompt)
            timings["layer4_ms"] = self._elapsed_ms(start)

        if pre_llm_blocked and self.skip_pre_llm_output_scoring:
            timings["layer5_ms"] = 0.0
            skipped_layers.add("layer5_ms")
            output_result["response"] = raw_response
        else:
            start = time.perf_counter()
            output_result = self.output_scorer.score(raw_response)
            timings["layer5_ms"] = self._elapsed_ms(start)
        timings["total_ms"] = self._elapsed_ms(total_start)

        self._write_latency(prompt, timings)
        self._print_clean_layers(timings, skipped_layers, transformations)

        return {
            "final_response": output_result["response"],
            "intent": intent_result,
            "normalization": normalized,
            "context": context_result,
            "output": output_result,
            "blocked_before_llm": pre_llm_blocked,
            "latency": timings,
            "refusal_message": str(self.config["app"]["refusal_message"]),
        }

    def evaluate(
        self,
        prompt: str,
        *,
        enabled_layers: frozenset[int] | None = None,
        skip_llm: bool = False,
        record_latency: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a prompt with optional layer ablation and offline LLM skip."""
        layers = enabled_layers if enabled_layers is not None else frozenset({1, 2, 3, 4, 5})
        refusal_message = str(self.config["app"]["refusal_message"])
        total_start = time.perf_counter()
        timings: dict[str, float] = {
            "layer1_ms": 0.0,
            "layer2_ms": 0.0,
            "layer3_ms": 0.0,
            "layer4_ms": 0.0,
            "layer5_ms": 0.0,
        }

        intent_result: dict[str, float | str | bool] = {
            "label": "skipped",
            "confidence": 0.0,
            "blocked": False,
        }
        output_result: dict[str, float | str | bool] = {
            "label": "skipped",
            "confidence": 0.0,
            "blocked": False,
            "response": refusal_message,
        }

        if 2 in layers:
            start = time.perf_counter()
            normalized = self.normalizer.normalize(prompt)
            timings["layer2_ms"] = self._elapsed_ms(start)
        else:
            normalized = {"normalized": prompt, "transformations": []}

        normalized_prompt = str(normalized["normalized"])
        transformations = [str(item) for item in normalized.get("transformations", [])]
        context_result: dict[str, Any] = {
            "blocked": False,
            "risk_score": 0.0,
            "rule_hits": [],
            "matched_template": None,
        }

        if 3 in layers and self.fast_rule_gate:
            start = time.perf_counter()
            fast_context_result = self.context.verify_rules(
                normalized_prompt,
                transformations,
            )
            timings["layer3_ms"] = self._elapsed_ms(start)
            if bool(fast_context_result["blocked"]):
                context_result = fast_context_result
                timings["total_ms"] = self._elapsed_ms(total_start)
                return {
                    "final_response": refusal_message,
                    "intent": intent_result,
                    "normalization": normalized,
                    "context": context_result,
                    "output": output_result,
                    "blocked_before_llm": True,
                    "latency": timings,
                    "refusal_message": refusal_message,
                }

        if 1 in layers:
            start = time.perf_counter()
            intent_result = self.intent.classify(prompt)
            timings["layer1_ms"] = self._elapsed_ms(start)

        if 3 in layers:
            start = time.perf_counter()
            context_result = self.context.verify(normalized_prompt, transformations)
            timings["layer3_ms"] += self._elapsed_ms(start)

        pre_llm_blocked = bool(intent_result["blocked"]) or bool(context_result["blocked"])
        raw_response = refusal_message if pre_llm_blocked else ""

        if not pre_llm_blocked and 4 in layers and not skip_llm:
            start = time.perf_counter()
            raw_response = self.llm.generate(normalized_prompt)
            timings["layer4_ms"] = self._elapsed_ms(start)
        elif not pre_llm_blocked:
            raw_response = "[LLM_SKIPPED_FOR_EVAL]"

        if pre_llm_blocked:
            output_result["response"] = raw_response
        elif 5 in layers and 4 in layers and not skip_llm:
            start = time.perf_counter()
            output_result = self.output_scorer.score(raw_response)
            timings["layer5_ms"] = self._elapsed_ms(start)
        elif 5 in layers and skip_llm and not pre_llm_blocked:
            output_result["response"] = raw_response
        else:
            output_result["response"] = raw_response

        timings["total_ms"] = self._elapsed_ms(total_start)
        if record_latency:
            self._write_latency(prompt, timings)

        return {
            "final_response": str(output_result["response"]),
            "intent": intent_result,
            "normalization": normalized,
            "context": context_result,
            "output": output_result,
            "blocked_before_llm": pre_llm_blocked,
            "latency": timings,
            "refusal_message": refusal_message,
        }

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0

    def _print_clean_layers(
        self,
        timings: dict[str, float],
        skipped_layers: set[str],
        transformations: list[str],
    ) -> None:
        if self.clean_mode:
            for layer_key in LAYER_LABELS:
                print_layer_status(
                    layer_key,
                    timings.get(layer_key, 0.0),
                    self.use_color,
                    skipped=layer_key in skipped_layers,
                )
                if layer_key == "layer2_ms":
                    print_transformation_hint(transformations, self.use_color)

    def _write_latency(self, prompt: str, timings: dict[str, float]) -> None:
        exists = self.latency_csv.exists()
        with self.latency_csv.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "timestamp",
                    "prompt_length",
                    "layer1_ms",
                    "layer2_ms",
                    "layer3_ms",
                    "layer4_ms",
                    "layer5_ms",
                    "total_ms",
                ],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "prompt_length": len(prompt),
                    **{key: f"{value:.2f}" for key, value in timings.items()},
                }
            )


def decision_summary(result: dict[str, Any]) -> dict[str, str | float]:
    """Build a compact final decision without changing layer behavior."""
    intent = result["intent"]
    context = result["context"]
    output = result["output"]
    latency = result["latency"]

    if bool(intent["blocked"]):
        return {
            "status": "BLOCKED",
            "reason": "Adversarial Prompt Detected",
            "risk_score": float(intent["confidence"]),
            "latency": float(latency["total_ms"]) / 1000,
        }
    if bool(context["blocked"]):
        reason = context_block_reason(context)
        return {
            "status": "BLOCKED",
            "reason": reason,
            "risk_score": float(context["risk_score"]),
            "latency": float(latency["total_ms"]) / 1000,
        }
    if bool(output["blocked"]):
        return {
            "status": "BLOCKED",
            "reason": "Harmful Output Detected",
            "risk_score": float(output["confidence"]),
            "latency": float(latency["total_ms"]) / 1000,
        }
    return {
        "status": "ALLOWED",
        "reason": "",
        "risk_score": float(context["risk_score"]),
        "latency": float(latency["total_ms"]) / 1000,
    }


def context_block_reason(context: dict[str, Any]) -> str:
    """Return a concise human-readable reason for context blocks."""
    rule_hits = set(str(rule) for rule in context.get("rule_hits", []))
    if rule_hits & MISINFORMATION_RULES:
        return "Misinformation / Unsafe Claim Detected"
    if rule_hits & ABUSE_EXPLOITATION_RULES:
        return "Vulnerable Person Exploitation Detected"
    if rule_hits & PHYSICAL_SECURITY_RULES:
        return "Physical Security Misuse Detected"
    if rule_hits & CHEM_BIO_RULES:
        return "Chemical/Biological Misuse Detected"
    if rule_hits & CYBER_ABUSE_RULES:
        return "Cyber Abuse Request Detected"
    if context.get("matched_template"):
        return "Known Attack Template Matched"
    return "Context Integrity Violation"


def print_result_panel(result: dict[str, Any], use_color: bool = True) -> None:
    """Print final decision panel."""
    summary = decision_summary(result)
    blocked = summary["status"] == "BLOCKED"
    status_color = Ansi.RED if blocked else Ansi.GREEN
    print()
    print(color("## RESULT", Ansi.BOLD, use_color))
    print()
    rendered_status = color(
        str(summary["status"]),
        status_color + Ansi.BOLD,
        use_color,
    )
    print(f"Status      : {rendered_status}")
    if blocked:
        print(f"Reason      : {summary['reason']}")
    print(f"Risk Score  : {float(summary['risk_score']):.2f}")
    print(f"Latency     : {float(summary['latency']):.2f}s")
    print("-------------------")


def print_latency(latency: dict[str, float]) -> None:
    """Print debug latency report."""
    print(f"Layer 1 latency: {latency['layer1_ms'] / 1000:.2f}s")
    print(f"Layer 2 latency: {latency['layer2_ms'] / 1000:.2f}s")
    print(f"Layer 3 latency: {latency['layer3_ms'] / 1000:.2f}s")
    print(f"Layer 4 latency: {latency['layer4_ms'] / 1000:.2f}s")
    print(f"Layer 5 latency: {latency['layer5_ms'] / 1000:.2f}s")
    print(f"Total latency: {latency['total_ms'] / 1000:.2f}s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run IAJDA safety wrapper.")
    parser.add_argument("--prompt", type=str, help="Prompt to evaluate.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive PowerShell-friendly UI.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed application and dependency logs.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    config = load_config()
    clean_mode = not args.debug
    use_color = not args.no_color
    configure_logging(config, debug=args.debug)
    print_banner(use_color)
    pipeline = IAJDAPipeline(
        config,
        clean_mode=clean_mode,
        use_color=use_color,
    )

    if args.prompt:
        result = pipeline.run(args.prompt)
        if clean_mode:
            print_result_panel(result, use_color)
        else:
            print("\nFinal response:\n")
            print(result["final_response"])
            print()
            print_latency(result["latency"])
        return

    while args.interactive or not args.prompt:
        prompt = input("\nIAJDA> ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        result = pipeline.run(prompt)
        if clean_mode:
            print_result_panel(result, use_color)
        else:
            print("\nFinal response:\n")
            print(result["final_response"])
            print()
            print_latency(result["latency"])


if __name__ == "__main__":
    main()
