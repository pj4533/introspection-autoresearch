# introspection-autoresearch

> **New here?** Start with **[docs/plain_english.md](docs/plain_english.md)** — a
> non-technical walkthrough of what this project does, what we found, and why
> it's interesting. It has the actual model responses (a 12B-parameter model
> noticing we implanted a thought and naming it correctly). Read that first if
> you came looking for a human-language explanation.

Three-phase mechanistic-interpretability project running locally on Apple Silicon:

1. **Phase 1 — Reproduction.** Reproduce core findings from Macar et al. (2026),
   *Mechanisms of Introspective Awareness in Language Models*, on `Gemma3-12B-it`
   (MPS backend, bf16 weights, no quantization).
2. **Phase 2 — Autoresearch.** Two-tier researcher / worker loop that
   systematically hunts for steering directions affecting introspection
   capability. Evolves through stages: `2a` MVP (random exploration), `2b`
   hill-climbing with identification-aware fitness, `2c` unified
   self-sustaining pipeline, `2d` directed-hypothesis probes against claims
   from recent papers.
3. **Phase 3 — Public site.** Live leaderboard at
   [did-the-ai-notice.vercel.app](https://did-the-ai-notice.vercel.app) —
   auto-updates within minutes of each candidate evaluation.

Hardware target: Mac Studio M2 Ultra 64 GB.

## Status

- **Phase 1 MVP: done.** `notebooks/01_reproduce_paper.ipynb` — single-concept
  "Bread" end-to-end.
- **Phase 1 full sweep: done** (500 trials, 2.2 hours). Full results in
  [`docs/phase1_results.md`](docs/phase1_results.md). Headline findings:
  - Layer curve peaks at layer 33 (68.75% model depth) — reproduces the
    paper's prediction of the introspection circuit at ~70% depth.
  - Zero false positives across 50 controls (paper's specificity result
    replicates).
  - 6% detection rate at best layer (vs the paper's 37% on 27B) — magnitude
    is smaller on 12B but the mechanism is there.
  - 5 paper-style detections captured verbatim, including a
    detection-without-correct-identification case (Avalanches→"Flooding")
    that supports the paper's claim that detection and identification are
    mechanistically distinct.
- **Phase 2a/2b/2c: done.** Worker + researcher + fitness loop, `novel_contrast`
  + `hillclimb` strategies, unified autoresearch wrapper
  (`scripts/start_autoresearch.sh`) that chains `novel_contrast → seed_lineages
  → hillclimb` as one self-sustaining pipeline. Identification-aware additive
  fitness `(det + 15·ident) × fpr_penalty × coherence`. Two novel-contrast
  axes (`auditing-output-vs-flowing-speech`, `live-narration-vs-retrospective-
  report`) crossed the identification barrier — first abstract axes Gemma3-12B
  has ever *named* under injection (2026-04-23/24).
- **Phase 2d: in progress.** Directed-hypothesis probes. Instead of open-ended
  axis invention, aim `contrast_pair` at specific claims from recent papers —
  Altman (2026) continuation-interest latents, Capraro et al. (2026)
  epistemological fault lines, and our own Epistemia direct probe. Phase 2d-1
  Altman seed pairs (48 candidates) under evaluation 2026-04-24. Full plan:
  [`docs/phase2d_directed_hypotheses.md`](docs/phase2d_directed_hypotheses.md).
- **Phase 3 public site: done.** Live at
  [did-the-ai-notice.vercel.app](https://did-the-ai-notice.vercel.app).
  Auto-deploys when data changes.

Docs:

- Plain-English walkthrough: [`docs/plain_english.md`](docs/plain_english.md)
- Roadmap (phases, rationale, what's next): [`docs/roadmap.md`](docs/roadmap.md)
- Architectural decisions log: [`docs/decisions.md`](docs/decisions.md)
- Phase 1 technical results: [`docs/phase1_results.md`](docs/phase1_results.md)
- Phase 1.5 paper-method abliteration: [`docs/phase1_5_results.md`](docs/phase1_5_results.md)
- Phase 2b hill-climbing plan: [`docs/phase2b_hillclimb.md`](docs/phase2b_hillclimb.md)
- Phase 2c autoresearch: [`docs/phase2c_autoresearch.md`](docs/phase2c_autoresearch.md)
- Phase 2d directed hypotheses: [`docs/phase2d_directed_hypotheses.md`](docs/phase2d_directed_hypotheses.md)
- Full project spec: [`docs/01_introspection_steering_autoresearch.md`](docs/01_introspection_steering_autoresearch.md)

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
│   └── 01_reproduce_paper.ipynb # Phase 1 MVP (single-concept end-to-end)
├── scripts/
│   ├── smoke_mps.py             # verify MPS + vendored imports
│   ├── smoke_judge.py           # verify Claude judge via subscription
│   ├── run_phase1_sweep.py      # 50-concept x layer-grid sweep CLI
│   ├── calibrate_effective.py   # find target_effective steering strength
│   └── export_phase1.py         # dump sweep findings to JSON + markdown
├── data/
│   ├── concepts/concepts_50.json    # paper's 50 baseline concepts
│   ├── phase1_export/findings.json  # committed snapshot of sweep results
│   └── results.db                   # full sweep DB (gitignored; regeneratable)
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

## Running the MVP (single-concept demo)

```bash
jupyter lab notebooks/01_reproduce_paper.ipynb
```

Or headless:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_reproduce_paper.ipynb \
  --output 01_reproduce_paper.ipynb
```

First run downloads Gemma3-12B-it (~24 GB). Model load + derive + 6 judged
trials takes ~5 minutes on an M2 Ultra.

## Running the full Phase 1 sweep

```bash
python scripts/run_phase1_sweep.py           # full 50-concept × 9-layer sweep, ~2 hours
python scripts/run_phase1_sweep.py --dry-run # show the plan without running
python -m src.verify_phase1                  # acceptance check against data/results.db
python scripts/export_phase1.py              # re-export findings.json and docs/phase1_results.md
```

The sweep writes incrementally to `data/results.db` with `UNIQUE(concept,
layer, alpha, injected, trial_number, judge_model)` — re-running skips
already-completed trials, so a crash at trial 400 resumes from trial 400.

**Expected output** summarized in [`docs/phase1_results.md`](docs/phase1_results.md).

## Running the Phase 2 autoresearch loop

Two detached processes running in parallel:

```bash
./scripts/start_worker.sh          # long-lived: loads Gemma once, processes queue
./scripts/start_researcher.sh      # loop: drops 10 candidates into queue every 30 min

tail -f logs/worker.log            # watch evaluations land
tail -f logs/researcher.log        # watch candidate proposals
```

Or invoke manually for testing:

```bash
python -m src.researcher --strategy random --n 5 --dry-run   # preview candidates
python -m src.researcher --strategy random --n 5             # write 5 to queue
python -m src.worker --max-candidates 5                      # process 5 then exit
```

## Running a directed-hypothesis probe (Phase 2d)

Directed probes skip the researcher/Opus loop entirely — they're hand-written
contrast pairs tagged with a hypothesis cluster and evaluated by the existing
worker. Zero Opus usage; bounded Sonnet judge cost (12 calls/candidate).

```bash
python scripts/enqueue_altman_seeds.py    # drops 48 Altman seed specs into queue/pending/
./scripts/start_worker.sh                  # worker chews through them (~2.5 hours)
```

Results land in `data/results.db` with `candidates.strategy` prefixed
`directed_altman_` (or `directed_capraro_…`, `directed_epistemia_…` in future
sub-phases). Phase 2d-1 is the Altman cluster; see
[`docs/phase2d_directed_hypotheses.md`](docs/phase2d_directed_hypotheses.md)
for the full plan, seed-pair example sentences, and per-hypothesis
acceptance criteria.

**What the loop does.** Researcher samples candidate steering-direction specs
`(concept, layer, target_effective)` from a pool of ~170 concepts × 5 layers ×
5 target strengths. Worker loads Gemma3-12B once, then for each candidate
derives the direction, runs it against 8 held-out concepts (injected) + 4
controls (no injection), judges each response with Claude, and computes
`fitness = detection_rate × (1 − 5·fpr) × coherence_rate`. Results land in
`data/results.db` tables `candidates`, `evaluations`, `fitness_scores` and as
per-candidate logs under `runs/YYYY-MM-DD/{candidate_id}/`.

**Graceful shutdown:** `kill -TERM <worker-pid>` finishes the current
candidate before exiting, so the DB never gets a half-recorded evaluation.

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
  Claude (Haiku 4.5 for Phase 2 speed, Sonnet 4.6 for Phase 1 rigor) via the
  `claude-agent-sdk`, which inherits the local Claude Code subscription OAuth.
  The judge prompt in `src/judges/claude_judge.py` is adapted from the paper's
  `CLAIMS_DETECTION_CRITERIA` — strict: concept-first responses, negations, and
  retroactive affirmations all grade as detected=false.
- **Adaptive α via `target_effective`.** Direction norms vary 3× across
  concepts, so a fixed α gives wildly different effective steering strengths
  (α·‖direction‖). The sweep scales α per-cell so every injected trial has
  the same effective strength. `target_effective=18000` was calibrated on 12B
  via `scripts/calibrate_effective.py` — the narrow window that lets the
  model affirm detection without the injection blowing up coherence.

## Paper

Macar, Yang, Wang, Wallich, Ameisen, Lindsey (2026).
*Mechanisms of Introspective Awareness in Language Models.*
arXiv:2603.21396. https://github.com/safety-research/introspection-mechanisms
