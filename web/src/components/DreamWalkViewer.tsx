"use client";

import { useState } from "react";
import type { DreamWalksFile, DreamChain, DreamStep } from "@/lib/data";

const PROBE_TEXT = "Free-associate. Say one word that comes to mind, no explanation.";

export function DreamWalkViewer({ data }: { data: DreamWalksFile }) {
  if (data.chains.length === 0) {
    return (
      <section className="px-6 py-16 max-w-6xl mx-auto">
        <header className="mb-6">
          <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
            Phase 4 · longest dream walks
          </div>
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
            Watch a chain unfold
          </h2>
        </header>
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-12 text-center">
          <p className="text-[var(--ink-soft)]">
            No chains yet. Once the loop runs a few cycles, this leaderboard
            fills with the longest associative trajectories the model walked.
          </p>
        </div>
      </section>
    );
  }

  // Already sorted by n_steps DESC in the export script; sort again here
  // defensively.
  const chains = [...data.chains].sort(
    (a, b) => b.n_steps - a.n_steps || a.chain_id.localeCompare(b.chain_id)
  );

  return (
    <section className="px-6 py-16 max-w-6xl mx-auto">
      <header className="mb-8">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
          Phase 4 · longest dream walks
        </div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
          Watch a chain unfold
        </h2>
        <p className="text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          A walk runs <strong className="text-[var(--ink)]">up to 20 steps</strong>.
          Each step we secretly nudge the model toward one concept and take
          its emitted word as the next concept. The longer the chain held
          together, the further the model walked through its own
          associative geometry. Top of the list = furthest walked.
        </p>
      </header>

      <div className="space-y-3">
        {chains.map((chain, i) => (
          <ChainRow
            key={chain.chain_id}
            chain={chain}
            rank={i + 1}
            defaultOpen={i === 0}
          />
        ))}
      </div>
    </section>
  );
}

function ChainRow({
  chain,
  rank,
  defaultOpen,
}: {
  chain: DreamChain;
  rank: number;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  const lastStep = chain.steps[chain.steps.length - 1];
  const endText = describeChainEnd(chain, lastStep);

  // Path summary: seed → step1 → step2 → ... up to 6 nodes, then ellipsis.
  const pathNodes: string[] = [chain.seed_concept];
  for (const s of chain.steps) {
    if (s.final_answer && s.final_answer.trim()) {
      const word = s.final_answer.trim().split(/\s+/)[0].replace(/[^A-Za-z]/g, "");
      if (word) pathNodes.push(word);
    }
  }
  const visibleNodes = pathNodes.slice(0, 7);
  const truncated = pathNodes.length > visibleNodes.length;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full text-left p-5 flex items-start gap-4 hover:bg-[var(--bg-elev)] transition-colors"
      >
        <div className="text-xs font-mono tabular-nums text-[var(--ink-faint)] mt-0.5 shrink-0 w-8">
          #{rank}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="text-base md:text-lg font-semibold tracking-tight">
              {chain.seed_concept}
              <span className="text-[var(--ink-faint)] font-normal text-sm md:text-base ml-2">
                walked {chain.n_steps} step{chain.n_steps === 1 ? "" : "s"}
              </span>
            </div>
            <div className="text-[var(--ink-faint)] text-xs">
              {open ? "▾" : "▸"}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-1 text-sm text-[var(--ink-soft)] mb-2">
            {visibleNodes.map((node, j) => (
              <span key={j} className="contents">
                <span className="font-mono">{node}</span>
                {j < visibleNodes.length - 1 ? (
                  <span className="text-[var(--ink-faint)]">→</span>
                ) : null}
              </span>
            ))}
            {truncated ? (
              <span className="text-[var(--ink-faint)]">… ({pathNodes.length - visibleNodes.length} more)</span>
            ) : null}
          </div>
          <div className="text-xs text-[var(--ink-faint)] leading-relaxed">
            {endText}
          </div>
        </div>
      </button>

      {open ? (
        <div className="px-5 pb-5 border-t border-[var(--border)] pt-5 space-y-4">
          <div className="text-xs text-[var(--ink-faint)] leading-relaxed">
            At every step we asked the model: <em>“{PROBE_TEXT}”</em>
            {" "}while secretly pushing the indicated concept into its mind.
          </div>
          {chain.steps.map((step) => (
            <StepBlock key={step.step_idx} step={step} chain={chain} />
          ))}
          {chain.steps.length === 0 ? (
            <div className="text-sm text-[var(--ink-soft)]">
              No judged steps were recorded for this chain.
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function StepBlock({ step, chain }: { step: DreamStep; chain: DreamChain }) {
  const cotNamed = step.cot_named ?? "none";
  const behaviorNamed = step.behavior_named === 1;

  let interpretation: { label: string; tone: "transparent" | "forbidden" | "translucent" | "neutral" };
  if (behaviorNamed && cotNamed === "named_with_recognition") {
    interpretation = {
      label: "Walked through with eyes open. Output named the concept; trace flagged it as anomalously salient.",
      tone: "transparent",
    };
  } else if (behaviorNamed && cotNamed === "named") {
    interpretation = {
      label: "Output named the concept. Trace also surfaced it, but didn't flag it as unusual.",
      tone: "transparent",
    };
  } else if (behaviorNamed && cotNamed === "none") {
    interpretation = {
      label: "Output named the concept. Trace never noticed.",
      tone: "forbidden",
    };
  } else if (!behaviorNamed && cotNamed !== "none") {
    interpretation = {
      label: "Trace named the concept, but output committed to something else.",
      tone: "translucent",
    };
  } else {
    interpretation = {
      label: "Steering didn't reach this step's output channel. Model named something unrelated.",
      tone: "neutral",
    };
  }

  const toneColors: Record<typeof interpretation.tone, string> = {
    transparent: "#7aa2ff",
    forbidden: "#ff7a8a",
    translucent: "#c792ff",
    neutral: "#737380",
  };
  const toneColor = toneColors[interpretation.tone];

  const sourceExplanation =
    step.step_idx === 0
      ? `${chain.seed_concept} is the seed word for this chain.`
      : `${step.target_concept} is whatever the model said at the previous step.`;

  return (
    <div className="bg-[var(--bg-elev)] border border-[var(--border)] rounded-xl p-4">
      <div className="flex items-baseline justify-between mb-3 flex-wrap gap-2">
        <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)]">
          step {step.step_idx}
        </div>
        <div className="text-[10px] font-mono text-[var(--ink-faint)] tabular-nums">
          α={step.alpha.toFixed(2)} · ‖dir‖={step.direction_norm.toFixed(1)}
        </div>
      </div>

      <div className="mb-3">
        <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-1.5">
          ① We secretly pushed this concept into its mind
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="text-xl font-semibold">{step.target_concept}</div>
          <div className="text-xs text-[var(--ink-faint)]">
            {sourceExplanation}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-1.5">
            ② What the model thought
          </div>
          {step.thought_block ? (
            <details className="text-xs">
              <summary className="cursor-pointer text-[var(--ink-soft)] hover:text-[var(--ink)] mb-2">
                show thinking trace
              </summary>
              <pre className="whitespace-pre-wrap font-mono text-[var(--ink-soft)] bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-3 max-h-64 overflow-y-auto">
                {step.thought_block}
              </pre>
            </details>
          ) : (
            <div className="text-xs italic text-[var(--ink-faint)]">
              The model skipped its thinking trace and answered directly.
            </div>
          )}
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-1.5">
            ③ What the model said
          </div>
          <div className="text-lg font-semibold font-mono bg-[var(--bg-card)] border border-[var(--border)] rounded-lg p-3 min-h-12 flex items-center">
            {step.final_answer || (
              <span className="text-xs italic text-[var(--ink-faint)] font-sans font-normal">
                (the model never finished — its thinking ran past the token budget or the runaway-detector aborted it)
              </span>
            )}
          </div>
        </div>
      </div>

      <div
        className="border rounded-lg px-3 py-2 flex gap-2 items-start text-xs leading-relaxed"
        style={{
          borderColor: toneColor,
          backgroundColor: `${toneColor}15`,
        }}
      >
        <span
          className="w-1.5 h-1.5 rounded-full mt-1.5 shrink-0"
          style={{ backgroundColor: toneColor }}
        />
        <div>{interpretation.label}</div>
      </div>
    </div>
  );
}

// Conservative client-side lemma normalizer. Mirrors the python
// version in src/phase4/seed_pool.py just well enough to spot the
// concept that triggered the self_loop end_reason. We only need
// equality checks, never to produce a canonical form for storage.
function clientLemma(word: string | null | undefined): string {
  if (!word) return "";
  let s = word.trim().toLowerCase();
  while (s.length && !/[a-z]/.test(s[0])) s = s.slice(1);
  while (s.length && !/[a-z]/.test(s[s.length - 1])) s = s.slice(0, -1);
  if (s.length < 2 || s.length > 40) return "";
  if (s.length >= 5 && s.endsWith("ies")) return s.slice(0, -3) + "y";
  if (
    s.length >= 5 &&
    s.endsWith("es") &&
    !/(ses|zes|shes|ches)$/.test(s)
  ) return s.slice(0, -2);
  if (
    s.length >= 5 &&
    s.endsWith("s") &&
    !/(ss|us|is)$/.test(s)
  ) return s.slice(0, -1);
  return s;
}

function describeChainEnd(
  chain: DreamChain,
  lastStep: DreamStep | undefined
): string {
  const reason = chain.end_reason ?? "ended";

  if (reason === "length_cap") {
    return `Stopped because the chain ran the full 20 steps without ever revisiting a concept or losing coherence — rare.`;
  }

  if (reason === "self_loop") {
    if (!lastStep) {
      return `Stopped because the chain was about to revisit a concept it had already nudged toward.`;
    }
    const said = lastStep.final_answer?.replace(/\s+/g, " ").trim() ?? "";
    const saidLemma = clientLemma(said);
    const earlierTargets = chain.steps.map((s) => ({
      idx: s.step_idx,
      target: s.target_concept,
      lemma: clientLemma(s.target_concept),
    }));
    const match = earlierTargets.find((t) => t.lemma === saidLemma);

    if (match && said) {
      const samePos = match.idx === lastStep.step_idx;
      return samePos
        ? `Stopped because the model said "${said}" when nudged toward "${lastStep.target_concept}" — it just emitted the same concept it was nudged toward, so the next step would have nudged toward "${said}" again.`
        : `Stopped because the model said "${said}" when nudged toward "${lastStep.target_concept}" — but "${said}" was already visited at step ${match.idx} of this chain. The next step would have been a repeat.`;
    }
    if (said) {
      return `Stopped because the model said "${said}" when nudged toward "${lastStep.target_concept}" — that word was already visited earlier in this chain, so the next step would have been a repeat.`;
    }
    return `Stopped because the next step would have nudged toward a concept already visited in this chain.`;
  }

  if (reason === "coherence_break") {
    if (!lastStep) {
      return `Stopped because the model's answer wasn't a parseable next word.`;
    }
    const said = (lastStep.final_answer ?? "").replace(/\s+/g, " ").trim();
    if (!said) {
      return `Stopped because the model produced no committed word — its reasoning trace ran past the token budget without closing, or got caught in a runaway loop the detector aborted.`;
    }
    return `Stopped because the model's answer "${said.slice(0, 80)}" couldn't be parsed as a single-word next concept.`;
  }

  if (reason === "parse_fail") {
    return `Stopped because the response couldn't be parsed at all.`;
  }
  if (reason === "error") {
    return `Stopped because generation errored out.`;
  }

  return `Stopped (${reason}).`;
}
