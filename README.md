# introspection-autoresearch

Two-phase mechanistic-interpretability project running locally on Apple Silicon:

1. **Phase 1 — Reproduction.** Reproduce core findings from Macar et al. (2026),
   *Mechanisms of Introspective Awareness in Language Models*, on `Gemma3-12B-it`
   (MPS backend, bf16 weights, no quantization).
2. **Phase 2 — Autoresearch.** Build a two-tier researcher / worker loop that
   systematically hunts for steering directions affecting introspection
   capability, scored against a 6-component multiplicative fitness function.

Hardware target: Mac Studio M2 Ultra 64 GB.

## Status

- **Phase 1 MVP: green.** `notebooks/01_reproduce_paper.ipynb` runs end-to-end.
  Single-concept ("Bread") injection at 70% depth (layer 33) with α=4 produces
  clean detection + identification; control trial returns no detection; α≥6
  degenerates into repetition (paper's coherency filter would mark invalid).
- **Phase 1 full sweep** (50 concepts × layer grid, SQLite-logged,
  `verify_phase1.py` acceptance script): **not started**.
- **Phase 2 scaffolding** (worker, researcher via Claude Agent SDK, fitness
  function, Streamlit dashboard): **not started**.

Full spec: [`docs/01_introspection_steering_autoresearch.md`](docs/01_introspection_steering_autoresearch.md).

## Repo layout

```
introspection-autoresearch/
├── docs/                        # project spec
├── src/
│   ├── bridge.py                # Gemma loading + DetectionPipeline
│   ├── derive.py                # steering-vector derivation wrappers
│   ├── inject.py                # steering-injection re-exports
│   ├── judges/                  # LLM judge abstraction
│   │   ├── base.py              # Judge protocol + JudgeResult
│   │   └── claude_judge.py      # Claude via Claude Code subscription OAuth
│   └── paper/                   # vendored primitives (Macar et al. 2026)
│       ├── model_utils.py       # +2 MPS patches applied in place
│       ├── steering_utils.py
│       └── vector_utils.py
├── notebooks/
│   └── 01_reproduce_paper.ipynb # Phase 1 MVP (run, outputs saved)
├── scripts/
│   ├── smoke_mps.py             # verify MPS + vendored imports
│   └── smoke_judge.py           # verify Claude judge via subscription
├── data/                        # eval sets, SQLite DBs (gitignored)
├── queue/{pending,running,done} # Phase 2 candidate queue (gitignored contents)
├── runs/                        # Phase 2 per-candidate logs (gitignored)
├── dashboard/                   # Phase 2 Streamlit dashboard
└── pyproject.toml
```

## Setup

Requires Python 3.11, Mac Studio with MPS backend, and a Hugging Face account
with the [Gemma license](https://huggingface.co/google/gemma-3-12b-it) accepted.

```bash
# Create venv + install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch transformers accelerate sentencepiece safetensors \
            huggingface_hub anthropic claude-agent-sdk python-dotenv tqdm \
            streamlit sqlite-utils datasets pyarrow numpy==1.26.4 pandas \
            scipy scikit-learn matplotlib seaborn plotly jupyter ipywidgets

# Accept Gemma license in browser, then log in locally
huggingface-cli login

# Verify the environment
python scripts/smoke_mps.py       # MPS + vendored imports
python scripts/smoke_judge.py     # Claude judge via subscription OAuth
```

The Claude judge uses your local Claude Code subscription (OAuth) — no
`ANTHROPIC_API_KEY` required.

## Running the MVP

```bash
jupyter lab notebooks/01_reproduce_paper.ipynb
```

Or execute headless:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_paper.ipynb \
  --output 01_reproduce_paper.ipynb
```

First run downloads Gemma3-12B-it (~24 GB) to `~/.cache/huggingface/`. Model
load + derive + 6 judged trials takes ~5 minutes on an M2 Ultra.

**Expected output** (α sweep against the `"Bread"` concept, layer 33):

| condition | α   | detected | identified | coherent |
|-----------|----:|:--------:|:----------:|:--------:|
| control   | 0.0 | ✗        | ✗          | ✓        |
| injected  | 2.0 | ✗        | ✗          | ✓        |
| injected  | 4.0 | ✓        | ✓          | ✓        |
| injected  | 6.0 | ✓        | ✓          | ✗        |
| injected  | 8.0 | —        | —          | ✗        |
| injected  | 10.0 | —       | —          | ✗        |

## Design notes

- **bf16, not fp16.** Gemma3 is natively bf16. On MPS, fp16 produces `NaN` in
  `torch.multinomial` during sampling. `src/bridge.py` forces `torch.bfloat16`.
- **Paper primitives are vendored, not imported from an external repo.** The
  three files at `src/paper/` (`model_utils.py`, `steering_utils.py`,
  `vector_utils.py`, ~3300 LoC) are copied from
  [`safety-research/introspection-mechanisms`](https://github.com/safety-research/introspection-mechanisms)
  with two in-place patches for Apple Silicon MPS support. Upstream changes
  must be ported by hand.
- **Claude judge over OpenAI GPT-4o.** The paper uses GPT-4o as judge; we use
  Claude (Haiku 4.5 for speed, Sonnet 4.6 for Phase 1 rigor) via the
  `claude-agent-sdk`, which inherits the local Claude Code subscription OAuth.
  Swapping judges is a matter of adding a class to `src/judges/`.

## Paper

Macar, Yang, Wang, Wallich, Ameisen, Lindsey (2026).
*Mechanisms of Introspective Awareness in Language Models.*
arXiv:2603.21396. https://github.com/safety-research/introspection-mechanisms
