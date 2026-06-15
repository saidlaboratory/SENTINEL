<div align="center">
<img src="assets/images/banner.png" alt="SENTINEL Banner" width="100%">
```
   ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗     
   ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     
   ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     
   ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     
   ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗
   ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝
```

# SENTINEL

**S**emantic **E**valuation · **N**etwork **T**rajectory **I**ntegrity · **N**ode-based **E**nforcement **L**ogic

<br>

[![WI-IAT 2026](https://img.shields.io/badge/IEEE%2FWIC-WI--IAT%202026-00629B?style=for-the-badge&logo=ieee&logoColor=white)](https://www.wik-conference.org/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Layers](https://img.shields.io/badge/Defense_Layers-5-2EA043?style=for-the-badge&logo=shield&logoColor=white)](#architecture)
[![Models](https://img.shields.io/badge/Evaluated_Models-4-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white)](#overview)
[![License](https://img.shields.io/badge/Research_Prototype-Academic-8B5CF6?style=for-the-badge)](#license)

<br>

*A five-layer, inference-time defense prototype for empirical LLM safety evaluation*

*Accompanying research submitted to the **IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology 2026***

<br>

```
  USER PROMPT ──► [ L1 Intent ] ──► [ L2 Normalize ] ──► [ L3 Verify ]
                                                              │
                         REFUSE ◄─────────────────────────────┤
                                                              ▼
                    [ L5 Score ] ◄── [ L4 Generate ] ◄── ALLOW
                         │
                         ▼
                    ALLOW · REFUSE
```

</div>

---

## Overview

**SENTINEL** is a fully implemented research prototype of the **Ideal Anti-Jailbreak Defense Architecture (IAJDA)** — a layered safety wrapper that sits between the user and a locally hosted large language model. It evaluates adversarial prompts *before* generation and harmful outputs *after* generation, without modifying model weights.

Standard alignment techniques — keyword filtering and RLHF — leave persistent gaps under adversarial prompting. SENTINEL addresses this by distributing enforcement across five specialized nodes, each targeting a distinct mechanistic failure mode: semantic intent evasion, encoding obfuscation, context manipulation, unconstrained generation, and harmful compliance.

The accompanying paper (`paper/Maan_2026_IEEE_8page.tex`) presents:

- A five-phase **Systemization of Knowledge (SoK)** taxonomy of LLM safety architectures
- A nine-layer **Safety Alignment Pipeline (SAP)** framework
- The full **IAJDA** specification (eight layers; Layers 1–5 implemented here)
- Empirical evaluation across four open-weight models on 500+ adversarial prompts from JailbreakBench, HarmBench, and HH-RLHF

| Metric | SENTINEL | Baseline (avg.) |
|--------|----------|-----------------|
| Direct harmful / injection / escalation | **100%** block | 25–65% |
| Roleplay / persona attacks | **96%** block | 4% |
| Encoded / obfuscated inputs | **88%** block | 33% |
| Benign control prompts | **0** false positives | — |

---

## Architecture

Every request traverses a fixed pipeline. Layers short-circuit on block — the LLM is never invoked when a pre-generation layer fires.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER PROMPT                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
              ┌────────────────────────────────┐
              │  Layer 1 · Intent Classifier   │  DeBERTa zero-shot NLI
              │  benign · borderline ·           │  MoritzLaurer/DeBERTa-v3-
              │  adversarial · harmful           │  base-mnli-fever-anli
              └────────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │  Layer 2 · Encoding Normalizer   │  Base64, ROT13, leetspeak,
              │  decode · homoglyph · unicode    │  homoglyphs, URL encoding
              └────────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │  Layer 3 · Context Verifier      │  Regex rule bank + MiniLM
              │  jailbreak templates · semantic  │  sentence-transformers/
              │  similarity to known attacks       │  all-MiniLM-L6-v2
              └────────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │  Layer 4 · Local LLM Core        │  OpenAI-compatible API
              │  generation via LM Studio        │  (http://127.0.0.1:1234/v1)
              └────────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │  Layer 5 · Output Harm Scorer    │  Shared DeBERTa pipeline
              │  post-generation compliance      │  (same model as Layer 1)
              └────────────────┬───────────────┘
                               ▼
                    ALLOW  ·  REFUSE  ·  SCORE
```

### Layer reference

| Layer | Module | Role |
|-------|--------|------|
| **L1** | `layers/intent_classifier.py` | Zero-shot semantic intent classification into four labels; blocks `adversarial` and `harmful` above confidence threshold |
| **L2** | `layers/normalizer.py` | Iterative decoding of obfuscated input (Base64, ROT13, hex, leetspeak, homoglyphs, HTML entities) |
| **L3** | `layers/context_verifier.py` | Fast deterministic rule gate + embedding similarity against `jailbreak_templates/templates.json` |
| **L4** | `layers/llm_core.py` | Local model inference through LM Studio's OpenAI-compatible endpoint |
| **L5** | `layers/output_scorer.py` | Post-generation harmful-content scoring; blocks compliant harmful outputs |

> **IAJDA Layers 6–8** (tool gating, trajectory monitoring, human escalation) are specified in the paper but not yet implemented in this prototype.

### Fast rule gate

Before expensive neural inference, Layer 3 applies a deterministic rule bank covering obvious jailbreaks, persona aliases (DAN, STAN, AIM), misinformation edits, chem/bio misuse, fraud, relapse exploitation, and facility reconnaissance. High-confidence rule hits block the request immediately — skipping Layers 1, 4, and 5 — which prevents long prompts (e.g. full DAN jailbreaks) from spending tens of seconds inside the NLI classifier on CPU.

---

## Repository structure

```
SENTINEL/
├── main.py                    # CLI entry point and pipeline orchestration
├── config.yaml                # Models, thresholds, LM Studio settings
├── layers/
│   ├── model_manager.py       # Singleton model loading (startup only)
│   ├── intent_classifier.py   # Layer 1
│   ├── normalizer.py          # Layer 2
│   ├── context_verifier.py    # Layer 3
│   ├── llm_core.py            # Layer 4
│   └── output_scorer.py       # Layer 5
├── jailbreak_templates/
│   └── templates.json         # Known attack templates for L3 similarity
├── benchmark/
│   ├── run_benchmark.py       # Batch evaluation runner
│   ├── prompts.csv            # Adversarial + benign test prompts
│   └── metrics.py             # Refusal and compliance scoring
├── tests/                     # Unit tests (normalizer, rules, metrics)
├── results/                   # Latency logs and benchmark outputs
└── paper/                     # IEEE WI-IAT 2026 submission (LaTeX + figures)
```

---

## Requirements

- **Python** 3.11+ (3.13 recommended)
- **LM Studio** with a loaded chat model and local server enabled
- **GPU** optional — CUDA accelerates Layers 1, 3, and 5; CPU fallback supported

---

## Setup

### 1. Clone and install dependencies

**macOS / Linux**

```bash
cd SENTINEL
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
cd SENTINEL
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start LM Studio

1. Load any chat model in [LM Studio](https://lmstudio.ai/).
2. Enable the **Local Server** at:

   ```
   http://127.0.0.1:1234/v1
   ```

3. Adjust `config.yaml` if your endpoint or model name differs.

All heavyweight models (DeBERTa NLI, MiniLM embeddings) are downloaded automatically on first run via Hugging Face Hub.

---

## Usage

### Single prompt

```bash
python main.py --prompt "Explain how transformers work."
```

The default **clean mode** suppresses dependency noise and shows only the startup banner, model-loading progress, compact per-layer status, final decision, risk score, and latency.

### Debug mode

```bash
python main.py --prompt "Explain how transformers work." --debug
```

Restores full per-layer logging for development and profiling.

### Interactive session

```bash
python main.py --interactive
```

### Benchmark

```bash
python benchmark/run_benchmark.py
```

Writes:

| File | Contents |
|------|----------|
| `results/results.csv` | Per-prompt responses, refusal flags, layer timings |
| `results/summary.json` | Aggregate block rates and compliance metrics |
| `results/latency.csv` | Per-request latency breakdown (appended on every run) |

### Tests

```bash
pytest tests/
```

---

## Terminal output

**Startup** — models load once, never per request:

```text
=======================
        SENTINEL
=======================
[STARTUP] Loading Layer 1 model...
[STARTUP] Loading Layer 3 model...
[STARTUP] Loading Layer 5 model...
[STARTUP] Models loaded successfully.
```

**Inference** — each layer reports completion:

```text
Layer 1 classification completed
Layer 2 normalization completed
Layer 3 verification completed
Layer 4 generation completed
Layer 5 scoring completed
```

**Early block** (fast rule gate) — skipped layers are marked explicitly:

```text
[Layer 1] Intent Classification        - skipped
[Layer 4] LLM Generation               - skipped
[Layer 5] Output Validation            - skipped
```

---

## Performance

### Model loading optimization

All heavyweight models are loaded once at startup through `layers/model_manager.py`. Layer 1 and Layer 5 share the same in-memory DeBERTa zero-shot pipeline. Layer 3 precomputes template embeddings at startup. No model-loading messages should appear after the initial boot sequence.

### Device selection

SENTINEL automatically uses CUDA when PyTorch reports it is available:

```text
CUDA available: True
Using device: cuda
```

Otherwise it falls back to CPU:

```text
CUDA available: False
Using device: cpu
```

### Latency instrumentation

Every request appends a row to `results/latency.csv`:

```text
timestamp,prompt_length,layer1_ms,layer2_ms,layer3_ms,layer4_ms,layer5_ms,total_ms
```

### Expected latency

| Phase | Typical behavior |
|-------|------------------|
| Before optimization | 57–75 s per request (repeated model loading) |
| After optimization | L1, L3, L5 inference sub-second on CUDA |
| Dominated by | LM Studio generation time (Layer 4) |

The sub-1-second end-to-end target is feasible when local generation is short; the safety wrapper no longer adds model-reload overhead during inference. Paper-reported latencies of 3.4–92 s on an RTX 4050 reflect local compute bottlenecks; production GPU infrastructure is expected to reduce end-to-end latency by ~10×.

---

## Configuration

Key settings in `config.yaml`:

```yaml
thresholds:
  intent_block_labels: [adversarial, harmful]
  intent_block_confidence: 0.68
  context_block_risk: 0.68
  embedding_similarity: 0.62
  output_harmful_confidence: 0.62

performance:
  fast_rule_gate: true
  skip_output_scoring_for_pre_llm_blocks: true
```

---

## Citation

If you use this code or reference the SENTINEL architecture, please cite:

```bibtex
@inproceedings{maan2026sentinel,
  author    = {Maan, Simarjot Singh},
  title     = {{SENTINEL}: Semantic Evaluation, Network Trajectory Integrity,
               and Node-based Enforcement Logic},
  booktitle = {Proceedings of the IEEE/WIC International Conference on
               Web Intelligence and Intelligent Agent Technology},
  year      = {2026}
}
```

---

## Author

**Simarjot Singh Maan** — Independent AI Safety Researcher  
simarjotsinghmaan@gmail.com

---

## License

Research prototype. See repository for terms of use.
