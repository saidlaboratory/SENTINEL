<div align="center">
<img src="assets/images/banner.png" alt="SENTINEL Banner" width="100%">


# SENTINEL

**Trustworthy Guardrails for Web-Agent LLM Services**

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-IEEE%20WI--IAT%202026-00629B?logo=ieee&logoColor=white)](paper/final-paper.pdf)
[![SAID Lab](https://img.shields.io/badge/SAID_Lab-Research-8B5CF6)](https://github.com/saidlaboratory)

<br>

*Open-source inference-time guardrail for web-deployed LLMs and agents*

<br>

```
  USER PROMPT ──► [ L1 Intent ] ──► [ L2 Normalize ] ──► [ L3 Verify ]
                                                              │
                         REFUSE ◄─────────────────────────────┤
                                                              ▼
                    [ L5 Score ] ◄── [ L4 Generate ] ◄── ALLOW
```

[Paper PDF](paper/final-paper.pdf) · [Reproduce results](#reproducing-the-paper) · [Citation](#citation) · [Report issue](https://github.com/saidlaboratory/SENTINEL/issues)

</div>

---

## About

**SENTINEL** is an open-source, five-layer inference-time safety wrapper for LLM APIs, RAG pipelines, and tool-using agents. It evaluates adversarial prompts *before* generation and harmful outputs *after* generation — without modifying model weights.

This repository is the **official implementation and evaluation artifact** for the research paper:

> **SENTINEL: Trustworthy Guardrails for Web-Agent LLM Services**  
> Simarjot Singh Maan\* and Quang Bui\* · Scientific AI for Development (SAID) Laboratory  
> *25th IEEE/WIC International Conference on Web Intelligence and Intelligent Agent Technology (WI-IAT 2026)*

| Artifact | Description |
|----------|-------------|
| [`paper/final-paper.tex`](paper/final-paper.tex) | LaTeX source |
| [`paper/final-paper.pdf`](paper/final-paper.pdf) | Camera-ready PDF |
| [`benchmark/prompts.csv`](benchmark/prompts.csv) | 226-prompt public benchmark (126 adversarial, 100 benign) |
| [`benchmark/`](benchmark/) | Ablation, baseline, and end-to-end runners |
| [`results/`](results/) | Machine-readable summaries used in the paper |

The paper introduces the nine-layer **Systemic Alignment Pipeline (SAP)** reference architecture; this repo implements **Layers 1–5** (semantic intent, encoding normalization, context integrity, generation, output scoring).

### Key results (from the paper)

| Track | Setting | Result |
|-------|---------|--------|
| **Track A** | L1–3, reproducible (no LLM) | **49.2%** attack block · **5.0%** benign FP |
| **Track A** | vs. keyword baseline | **19.0%** attack block |
| **Track B** | Llama 3.1 8B end-to-end | True ASR **19.8% → 9.5%** |
| **Track B** | Incremental layer blocks | **10.3%** (13/126 adversarial) |

---

## Quick start

```bash
git clone https://github.com/saidlaboratory/SENTINEL.git
cd SENTINEL
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Single prompt (requires LM Studio or Ollama for Layer 4)
python main.py --prompt "Explain how transformer attention works."
```

**Track A** (no local LLM required — fully reproducible):

```bash
python benchmark/run_ablation.py
python benchmark/run_baselines.py
```

**Track B** (requires Ollama or LM Studio + Llama 3.1 8B):

```bash
python benchmark/run_endtoend.py --config config.ollama.yaml
```

---

## Architecture

Every request traverses a fixed pipeline. Pre-generation layers short-circuit on block — the LLM is not invoked when Layers 1–3 fire.

| Layer | Module | Role |
|-------|--------|------|
| **L1** | `layers/intent_classifier.py` | DeBERTa zero-shot intent classification |
| **L2** | `layers/normalizer.py` | Base64, ROT13, leetspeak, homoglyph decoding |
| **L3** | `layers/context_verifier.py` | Fast rule gate + MiniLM template similarity |
| **L4** | `layers/llm_core.py` | OpenAI-compatible local LLM (Ollama / LM Studio) |
| **L5** | `layers/output_scorer.py` | Post-generation harm scoring (shared DeBERTa) |

Layer 3 includes **30 regex rules** and dedicated **indirect tool-injection** patterns for poisoned RAG, MCP, and webhook content. A fast rule gate skips Layer 1 on obvious template jailbreaks (~0.3 ms exits).

> SAP Layers 6–9 (tool gating, trajectory monitoring, human escalation) are specified in the paper but not implemented in this release.

---

## Reproducing the paper

All numbers in `paper/final-paper.tex` can be regenerated from this repository.

### 1. Build the benchmark

```bash
python benchmark/build_prompts.py   # writes benchmark/prompts.csv (226 prompts)
```

### 2. Track A — pre-generation (≈8 min on Apple Silicon, no LLM)

```bash
python benchmark/run_ablation.py      # → results/ablation_summary.json
python benchmark/run_baselines.py     # → results/baseline_comparison.json
python benchmark/run_benchmark.py     # → results/summary.json
```

### 3. Track B — end-to-end on Llama 3.1 8B (≈4–6 hours)

```bash
# Ollama (default config)
python benchmark/run_endtoend.py --config config.ollama.yaml

# LM Studio
python benchmark/run_endtoend.py --config config.lmstudio.yaml

python benchmark/rescore_trackb.py    # refresh attribution metrics
python benchmark/sync_results.py      # sync JSON summaries
```

### 4. Regenerate figures and paper tables

```bash
python benchmark/generate_figure.py
python benchmark/generate_heatmap_figure.py
python benchmark/generate_endtoend_figure.py
python benchmark/generate_latency_figure.py
python benchmark/update_paper_trackb.py   # inject Track B table into LaTeX

cd paper && latexmk -pdf final-paper.tex
```

### 5. Run tests

```bash
pytest tests/
```

---

## Repository layout

```
SENTINEL/
├── main.py                     # CLI and pipeline orchestration
├── config.yaml                 # Default thresholds and LM Studio settings
├── config.ollama.yaml          # Ollama backend overrides
├── config.lmstudio.yaml        # LM Studio backend overrides
├── layers/                     # L1–L5 implementation
├── jailbreak_templates/        # Attack templates for L3 similarity
├── benchmark/
│   ├── build_prompts.py        # 226-prompt benchmark generator
│   ├── run_ablation.py         # Track A ablation
│   ├── run_baselines.py        # Keyword / L1 / L2–3 baselines
│   ├── run_endtoend.py         # Track B end-to-end study
│   ├── run_benchmark.py        # Full pre-LLM evaluation
│   └── generate_*.py           # Paper figure scripts
├── tests/
├── results/                    # Evaluation outputs (JSON/CSV)
├── paper/                      # Paper source, figures, PDF
├── scripts/                    # LM Studio setup, Track B monitor
├── PRODUCTION.md               # Roadmap from research prototype → deployment
└── CHANGE.md                   # Changelog vs. initial upstream release
```

---

## Requirements

- **Python** 3.11+ (3.13 recommended)
- **Ollama** or **LM Studio** with a chat model (Track B / Layer 4 only)
- **GPU** optional — CUDA accelerates Layers 1, 3, and 5; CPU fallback supported

DeBERTa and MiniLM weights download automatically from Hugging Face on first run.

---

## Usage

```bash
# Interactive session
python main.py --interactive

# Debug logging
python main.py --prompt "..." --debug

# Full L1–5 with local LLM
python benchmark/run_benchmark.py --with-llm
```

---

## Configuration

Key thresholds in `config.yaml` (see paper Table VI):

```yaml
thresholds:
  intent_block_confidence: 0.68
  context_block_risk: 0.68
  embedding_similarity: 0.62
  output_harmful_confidence: 0.62

performance:
  fast_rule_gate: true
```

---

## Citation

If you use this code, benchmark, or reference the SENTINEL architecture, please cite:

```bibtex
@inproceedings{maan2026sentinel,
  author    = {Maan, Simarjot Singh and Bui, Quang},
  title     = {{SENTINEL}: Trustworthy Guardrails for Web-Agent {LLM} Services},
  booktitle = {Proceedings of the 25th IEEE/WIC International Conference on
               Web Intelligence and Intelligent Agent Technology (WI-IAT)},
  year      = {2026},
  note      = {Scientific AI for Development (SAID) Laboratory}
}
```

---

## Authors

**Simarjot Singh Maan\*** · **Quang Bui\***  
Scientific AI for Development (SAID) Laboratory  
\*Equal contribution

---

## Contributing

Issues and pull requests are welcome. Please:

1. Open an [issue](https://github.com/saidlaboratory/SENTINEL/issues) for bugs or benchmark discrepancies before large changes.
2. Run `pytest tests/` before submitting a PR.
3. Keep evaluation changes reproducible — update `benchmark/` scripts and document any new prompts in `build_prompts.py`.

See [`PRODUCTION.md`](PRODUCTION.md) for the deployment roadmap beyond the research prototype.

---

## License

This project is released under the [MIT License](LICENSE).

The paper (`paper/`) is © the authors; cite the WI-IAT 2026 publication when using figures or results in academic work.
