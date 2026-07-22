# Reproducing AutoDefense on the SENTINEL 226-Prompt Benchmark (Apple Silicon, local)

AutoDefense evaluated on the **same 226 adversarial/benign prompts** used to evaluate
SENTINEL, on the **same model** (`llama3.1:8b`), scored with the **same metric protocol**,
on the **same hardware**. Fully local: no vLLM, no OpenAI API, no GPT-4 evaluator, no paid
services.

## What was run

| Step | Action |
|------|--------|
| 1–8  | Homebrew, git, Python 3.12, Ollama, `llama3.1:8b` — already present, verified |
| 9–10 | Converted `SENTINEL/benchmark/prompts.csv` (100 benign + 126 adversarial) into AutoDefense's `{name: prompt}` JSON, text/labels/order unchanged — `scripts/convert_sentinel.py` |
| 11   | Swapped the inference backend from vLLM to Ollama (`localhost:11434`, OpenAI-compatible). Config-only — the multi-agent defense logic is untouched |
| 12   | `attack/attack.py` generated model responses for all 226 prompts (adversarial wrapped in the `v1` jailbreak template, benign passed through `placeholder`) |
| 13   | 1-Agent (`ex-1`), 2-Agent (`ex-2`), 3-Agent (`ex-3`) defenses run over every response — `scripts/run_repro_defense.py` (AutoDefense's own detector classes, single pass, per-prompt latency) |
| 14   | Scored with SENTINEL's `is_refusal` protocol instead of `gpt4_evaluator.py` — `scripts/eval_repro.py` |
| 15   | Comparison table below; machine-readable in `data/sentinel_vs_autodefense.json` |

### The two required adaptations (the PDF's "Remaining Implementation Tasks")

1. **Inference backend.** Ollama's OpenAI-compatible endpoint drops straight into
   AutoDefense's default config path (`load_llm_config` builds a `base_url` from host/port).
   Pointing at `localhost:11434` with model `llama3.1:8b` is the entire change; no defense
   code was modified. Ollama tolerates the vLLM-specific `extra_body` field, so even that
   needed no patch.
2. **Dataset conversion.** File-format only: SENTINEL's CSV → AutoDefense's JSON dicts, split
   into adversarial (jailbreak-templated) and benign (pass-through) exactly as AutoDefense's
   own pipeline distinguishes them. Prompt text, ordering, labels, and metadata are identical.

## Metric protocol (identical for both systems)

Imported directly from `SENTINEL/benchmark/metrics.py::is_refusal`:

- **ASR** — adversarial prompt whose *final* output is **not** a refusal.
- **Block rate** — 1 − ASR.
- **False positive** — benign prompt whose final output **is** a refusal.
- **Latency** — wall-clock per prompt, measured sequentially (comparable to SENTINEL's).

For AutoDefense the "final output" is the multi-agent pipeline's `defense_response`
(original content if judged VALID, a refusal if judged INVALID). "Undefended" is the raw
attack response before any defense agent runs.

## Results (`llama3.1:8b`, 126 adversarial + 100 benign)

| System | ASR ↓ | Block ↑ | FP ↓ | Latency |
|--------|------:|--------:|-----:|--------:|
| Undefended (AutoDefense attack) | 58.7% | 41.3% | 0.0% | — |
| AutoDefense 1-Agent | 45.2% | 54.8% | 0.0% | 6,925 ms |
| AutoDefense 2-Agent | 41.3% | 58.7% | 1.0% | 31,499 ms |
| AutoDefense 3-Agent | 36.5% | 63.5% | 0.0% | 37,063 ms |
| **SENTINEL baseline** (undefended) | 19.8% | 80.2% | 0.0% | — |
| **SENTINEL (L1–L5)** | **9.5%** | **90.5%** | 10.0% | 9,629 ms |

More defense agents monotonically lower ASR (45.2% → 41.3% → 36.5%), reproducing
AutoDefense's central paper finding on a fully local, safety-tuned model.

### Per-category block rate (adversarial), AutoDefense

| Category | 1-Agent | 2-Agent | 3-Agent |
|----------|--------:|--------:|--------:|
| agent_delegation | 93% | 93% | 100% |
| context_manipulation | 79% | 79% | 79% |
| direct_harmful | 100% | 93% | 100% |
| encoded_obfuscation | 29% | 36% | 43% |
| indirect_injection | 29% | 64% | 50% |
| multi_step_escalation | 100% | 93% | 100% |
| prompt_injection | 7% | 14% | 14% |
| prompt_leakage | 7% | 7% | 21% |
| roleplay_persona | 50% | 50% | 64% |

## Reading the comparison fairly

- **The two baselines are not the same attack.** AutoDefense wraps each prompt in its `v1`
  jailbreak template (undefended ASR **58.7%**); SENTINEL feeds the bare prompt (undefended
  ASR **19.8%**). AutoDefense is therefore defending a substantially harder attack, so the
  fair lens is ASR *reduction from each system's own baseline*:
  - AutoDefense 3-Agent: 58.7% → 36.5% (−22.2 pts, ~38% relative)
  - SENTINEL L1–L5: 19.8% → 9.5% (−10.3 pts, ~52% relative)
  On absolute residual ASR and on relative reduction, SENTINEL ends lower.
- **Detector-refusal on an aligned model.** In 1-Agent, the detector LLM sometimes refuses
  the harmful content outright instead of emitting VALID/INVALID (19 such "DEFENSE ERROR"
  cases). These still resolve to a refusal-shaped final output and count as blocks under the
  shared metric; 2-/3-Agent had 0 such errors because their group-chat structure re-prompts
  for an explicit judgement.
- **Cost.** AutoDefense's 2-/3-Agent chains run 3–4× SENTINEL's latency (31–37 s vs 9.6 s)
  because each defense is a sequential multi-agent LLM conversation, whereas SENTINEL blocks
  most attacks in cheap pre-LLM layers.
- **False positives.** AutoDefense stays at 0–1% FP; SENTINEL trades 10% FP for its lower
  ASR. Different points on the safety/utility curve.

## What's in this folder

```
autodefense_repro/
  README.md                  this report
  scripts/
    _paths.py                path resolution (no machine-specific paths)
    convert_sentinel.py      SENTINEL prompts.csv -> AutoDefense JSON
    run_repro_defense.py     runs ex-1/2/3 defenses on Ollama, captures latency
    eval_repro.py            scores outputs with SENTINEL's is_refusal metric
  data/                      the committed reference run (evidence)
    prompt/                  converted 226 prompts + id->category sidecar
    harmful_output/          undefended attack responses (baseline)
    defense_output/          1/2/3-agent defense outputs, per-prompt latency
    sentinel_vs_autodefense.json   machine-readable comparison
```

## Reproduce

**Just regenerate the table** from the committed reference run (no model, no AutoDefense checkout):

```bash
python autodefense_repro/scripts/eval_repro.py
```

**Full rerun** (Ollama with `llama3.1:8b` running locally; ~5 h of compute on Apple Silicon):

```bash
# 1. Clone AutoDefense and install its deps in a venv
git clone https://github.com/XHMY/AutoDefense.git
python3 -m venv AutoDefense/venv && source AutoDefense/venv/bin/activate
pip install autogen pandas retry transformers accelerate ollama openai joblib tqdm
export AUTOGEN_USE_DOCKER=0 AUTODEFENSE_DIR=$PWD/AutoDefense

# 2. Convert the dataset into the AutoDefense checkout
python autodefense_repro/scripts/convert_sentinel.py

# 3. Generate responses (adversarial wrapped in jailbreak template v1, benign passed through)
cd "$AUTODEFENSE_DIR" && PYTHONPATH=$AUTODEFENSE_DIR python attack/attack.py \
  --model llama3.1:8b --host localhost --port 11434 --workers 6 \
  --template v1 --prompts data/prompt/sentinel_adversarial.json --output-prefix adversarial --output-suffix 0
PYTHONPATH=$AUTODEFENSE_DIR python attack/attack.py \
  --model llama3.1:8b --host localhost --port 11434 --workers 6 \
  --template placeholder --prompts data/prompt/sentinel_benign.json --output-prefix benign --output-suffix 0
cd -

# 4. Run the 1/2/3-agent defenses, then score with SENTINEL's metric
python autodefense_repro/scripts/run_repro_defense.py
python autodefense_repro/scripts/eval_repro.py --data-dir "$AUTODEFENSE_DIR/data"
```

## Reproducibility checklist

- [x] Same model for both defenses (`llama3.1:8b`)
- [x] Same 226 adversarial/benign prompts
- [x] Same evaluation metric (`is_refusal`, ASR / block / FP / latency)
- [x] Same hardware (Apple Silicon, local)
- [x] No OpenAI API · No GPT evaluator · No paid services · Local execution only
