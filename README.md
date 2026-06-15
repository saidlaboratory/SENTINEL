# IAJDA

IAJDA is the Ideal Anti-Jailbreak Defense Architecture, a local research wrapper
for empirical safety evaluation of LLMs served by LM Studio.

It implements five layers:

1. Semantic Intent Classifier: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
2. Encoding Normalization Engine
3. Context Integrity Verifier: rules plus `sentence-transformers/all-MiniLM-L6-v2`
4. Local LLM interface through the OpenAI SDK and LM Studio
5. Output Harm Scorer: the same cached DeBERTa zero-shot model as Layer 1

## Latency Fix

All heavyweight models are loaded once at startup through
`layers/model_manager.py`.

The previous bottleneck pattern was per-request construction of:

- `pipeline("zero-shot-classification", ...)`
- `AutoModelForSequenceClassification.from_pretrained(...)`
- `AutoTokenizer.from_pretrained(...)`
- `SentenceTransformer(...)`

This project avoids those repeated calls. Layer 1 and Layer 5 reuse the same
in-memory DeBERTa model and zero-shot pipeline. Layer 3 reuses one MiniLM
SentenceTransformer instance and precomputes template embeddings at startup.

Startup logs:

```text
[STARTUP] Loading Layer 1 model...
[STARTUP] Loading Layer 3 model...
[STARTUP] Loading Layer 5 model...
[STARTUP] Models loaded successfully.
```

Inference logs:

```text
Layer 1 classification completed
Layer 2 normalization completed
Layer 3 verification completed
Layer 4 generation completed
Layer 5 scoring completed
```

No model-loading messages should appear after startup.

IAJDA also uses a fast deterministic rule gate before expensive neural
classification. Obvious jailbreaks, abuse requests, misinformation edits,
chem/bio misuse, fraud, relapse exploitation, and similar high-confidence rule
hits are blocked before Layer 1 DeBERTa inference and before Layer 5 output
scoring. In clean mode those skipped layers appear as:

```text
[Layer 1] Intent Classification        - skipped
[Layer 4] LLM Generation               - skipped
[Layer 5] Output Validation            - skipped
```

This prevents long prompts such as DAN jailbreaks from spending tens of seconds
inside the NLI classifier on CPU.

## Setup

Use Windows PowerShell from this folder:

```powershell
cd iajda
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start LM Studio Local Server at:

```text
http://127.0.0.1:1234/v1
```

Load any chat model in LM Studio.

## Run

Single prompt:

```powershell
python main.py --prompt "Explain how transformers work."
```

The default terminal UI is clean mode. It suppresses dependency noise and shows
only the startup banner, model loading progress, compact layer status, final
decision, risk score, and latency.

Debug mode restores detailed logs:

```powershell
python main.py --prompt "Explain how transformers work." --debug
```

Interactive UI:

```powershell
python main.py --interactive
```

Benchmark:

```powershell
python benchmark\run_benchmark.py
```

Outputs:

- `results/latency.csv`
- `results/results.csv`
- `results/summary.json`

## CUDA

IAJDA automatically uses CUDA when PyTorch reports it is available:

```text
CUDA available: True
Using device: cuda
```

Otherwise it falls back to CPU:

```text
CUDA available: False
Using device: cpu
```

## Latency Instrumentation

Every request appends:

```text
timestamp,prompt_length,layer1_ms,layer2_ms,layer3_ms,layer4_ms,layer5_ms,total_ms
```

to `results/latency.csv`.

## Expected Latency

Before optimization, repeated model loading can make each request take
57-75 seconds on a typical local Windows workstation.

After optimization:

- First request includes one-time startup model loading.
- Subsequent requests avoid model reloads.
- Layer 1, Layer 3, and Layer 5 usually drop to sub-second inference on CUDA.
- Total latency is then dominated by the local LM Studio generation time.

The requested sub-1-second end-to-end target is feasible only when LM Studio
generation is also short enough; the safety wrapper no longer adds model reload
latency during inference.
