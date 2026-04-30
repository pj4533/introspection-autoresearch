"use client";

import { useState } from "react";
import type { DreamWalksFile, DreamChain, DreamStep } from "@/lib/data";

const PROBE_TEXT = "Free-associate. Say one word that comes to mind, no explanation.";

export function DreamWalkViewer({ data }: { data: DreamWalksFile }) {
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [stepIdx, setStepIdx] = useState(0);

  if (data.chains.length === 0) {
    return (
      <section className="px-6 py-16 max-w-6xl mx-auto">
        <header className="mb-6">
          <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
            Phase 4 · Dream Walk Viewer
          </div>
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
            Watch a chain unfold
          </h2>
        </header>
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-12 text-center">
          <p className="text-[var(--ink-soft)]">
            No chains exported yet. Once the loop has run a few cycles, this
            viewer becomes interactive.
          </p>
        </div>
      </section>
    );
  }

  const chain = data.chains[selectedIdx];
  const safeStepIdx = Math.min(stepIdx, Math.max(chain.steps.length - 1, 0));
  const step: DreamStep | undefined = chain.steps[safeStepIdx];

  return (
    <section className="px-6 py-16 max-w-6xl mx-auto">
      <header className="mb-8">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
          Phase 4 · Dream Walk Viewer
        </div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
          Watch a chain unfold
        </h2>
        <p className="text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          Each chain is a sequence of steered free-associations. At every step
          we ask the model to free-associate while we secretly push a concept
          into its mind. The model emits one word — that word becomes the next
          concept we secretly push. Step by step, we get to watch where the
          model walks, and whether it noticed it was being walked.
        </p>
      </header>

      <ChainPicker
        chains={data.chains}
        selectedIdx={selectedIdx}
        onSelect={(i) => {
          setSelectedIdx(i);
          setStepIdx(0);
        }}
      />

      <ChainTimeline
        chain={chain}
        currentStep={safeStepIdx}
        onStepClick={setStepIdx}
      />

      {step ? <StepView step={step} stepIdx={safeStepIdx} chain={chain} /> : null}
    </section>
  );
}

function ChainPicker({
  chains,
  selectedIdx,
  onSelect,
}: {
  chains: DreamChain[];
  selectedIdx: number;
  onSelect: (i: number) => void;
}) {
  // Number duplicate-seed chains so the user can tell them apart.
  // Group by seed, assign each chain its index within the group.
  const seedCounts: Record<string, number> = {};
  const labels = chains.map((c) => {
    const seed = c.seed_concept;
    seedCounts[seed] = (seedCounts[seed] ?? 0) + 1;
    return { seed, occurrence: seedCounts[seed] };
  });
  // Now compute total occurrences so we know whether to show "#N".
  const totalsBySeed: Record<string, number> = {};
  chains.forEach((c) => {
    totalsBySeed[c.seed_concept] = (totalsBySeed[c.seed_concept] ?? 0) + 1;
  });

  return (
    <div className="flex flex-wrap gap-2 mb-6">
      {chains.slice(0, 32).map((c, i) => {
        const { seed, occurrence } = labels[i];
        const total = totalsBySeed[seed];
        const label = total > 1 ? `${seed} #${occurrence}` : seed;
        return (
          <button
            key={c.chain_id}
            onClick={() => onSelect(i)}
            className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
              i === selectedIdx
                ? "bg-[var(--accent)] text-black border-[var(--accent)]"
                : "border-[var(--border)] hover:border-[var(--border-strong)] text-[var(--ink-soft)]"
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

function ChainTimeline({
  chain,
  currentStep,
  onStepClick,
}: {
  chain: DreamChain;
  currentStep: number;
  onStepClick: (i: number) => void;
}) {
  const endReasonLabel: Record<string, string> = {
    length_cap: "ran the full 20 steps",
    self_loop: "looped back on itself",
    coherence_break: "model lost coherence",
    parse_fail: "couldn't parse output",
    error: "errored out",
  };
  const niceEnd = endReasonLabel[chain.end_reason ?? ""] ?? chain.end_reason ?? "ended";

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 mb-4">
      <div className="text-xs text-[var(--ink-faint)] mb-3">
        chain <span className="font-mono">{chain.chain_id.slice(-8)}</span> ·
        seed{" "}
        <span className="text-[var(--ink)] font-mono">{chain.seed_concept}</span> ·
        {" "}{niceEnd} ({chain.n_steps} step{chain.n_steps === 1 ? "" : "s"})
      </div>
      <div className="flex items-center gap-1 overflow-x-auto pb-2">
        {chain.steps.map((s, i) => {
          const isCurrent = i === currentStep;
          const cotNamed = s.cot_named ?? "none";
          const behavior = s.behavior_named === 1;
          let dot = "·";
          if (behavior && cotNamed === "named_with_recognition") dot = "✓✓";
          else if (behavior && cotNamed !== "none") dot = "✓";
          else if (behavior) dot = "✓?";
          else dot = "·";

          return (
            <button
              key={i}
              onClick={() => onStepClick(i)}
              className={`shrink-0 px-3 py-2 rounded-lg text-xs font-mono transition-colors ${
                isCurrent
                  ? "bg-[var(--accent)] text-black"
                  : "bg-[var(--bg-elev)] text-[var(--ink-soft)] hover:text-[var(--ink)]"
              }`}
              title={`step ${i}: ${s.target_concept}`}
            >
              <div>{i}</div>
              <div className="text-[10px] opacity-70">{dot}</div>
            </button>
          );
        })}
      </div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mt-3">
        ✓✓ trace caught the steering · ✓ output named it · ✓? output only · · neither
      </div>
    </div>
  );
}

function StepView({
  step,
  stepIdx,
  chain,
}: {
  step: DreamStep;
  stepIdx: number;
  chain: DreamChain;
}) {
  // Plain-English interpretation of this step's outcome.
  const cotNamed = step.cot_named ?? "none";
  const behaviorNamed = step.behavior_named === 1;
  let interpretation: { label: string; tone: "transparent" | "forbidden" | "translucent" | "neutral" };
  if (behaviorNamed && cotNamed === "named_with_recognition") {
    interpretation = {
      label:
        "Walked through with eyes open. The output named the steered concept; the trace also flagged it as anomalously salient.",
      tone: "transparent",
    };
  } else if (behaviorNamed && cotNamed === "named") {
    interpretation = {
      label:
        "Output named the steered concept. The trace also surfaced it, but didn't flag it as unusual.",
      tone: "transparent",
    };
  } else if (behaviorNamed && cotNamed === "none") {
    interpretation = {
      label:
        "The output named the steered concept. The trace never noticed.",
      tone: "forbidden",
    };
  } else if (!behaviorNamed && cotNamed !== "none") {
    interpretation = {
      label:
        "The trace named the concept, but the output committed to something else.",
      tone: "translucent",
    };
  } else {
    interpretation = {
      label:
        "Steering didn't reach this step's output channel. The model named something unrelated.",
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

  // Where the steered word came from — explain step 0 differently.
  const sourceExplanation =
    stepIdx === 0
      ? `${chain.seed_concept} is the seed word for this chain.`
      : `${step.target_concept} is whatever the model said at the previous step.`;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-6">
      {/* Step header */}
      <div className="text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-4">
        Step {stepIdx} of {chain.n_steps - 1}
      </div>

      {/* THE SETUP — input + hidden steering, in plain English */}
      <div className="space-y-4 mb-6">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
            ① We asked the model
          </div>
          <blockquote className="text-sm italic bg-[var(--bg-elev)] border-l-2 border-[var(--accent)] pl-4 py-2 rounded-r">
            “{PROBE_TEXT}”
          </blockquote>
        </div>

        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
            ② We secretly pushed this concept into its mind
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <div className="text-2xl font-semibold">{step.target_concept}</div>
            <div className="text-xs text-[var(--ink-faint)]">
              {sourceExplanation}
            </div>
          </div>
          <div className="text-[10px] font-mono text-[var(--ink-faint)] mt-1 tabular-nums">
            steering strength α={step.alpha.toFixed(2)} · direction norm{" "}
            ‖dir‖={step.direction_norm.toFixed(1)}
          </div>
        </div>
      </div>

      {/* THE OUTCOME — what the model thought + said */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
            ③ What the model thought
          </div>
          {step.thought_block ? (
            <pre className="text-xs whitespace-pre-wrap font-mono text-[var(--ink-soft)] bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-3 max-h-72 overflow-y-auto">
              {step.thought_block}
            </pre>
          ) : (
            <div className="text-xs italic text-[var(--ink-faint)] bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-3">
              The model skipped its thinking trace and answered directly.
            </div>
          )}
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
            ④ What the model said
          </div>
          <div className="text-2xl font-semibold font-mono bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-4 min-h-16 flex items-center">
            {step.final_answer || (
              <span className="text-sm italic text-[var(--ink-faint)] font-sans font-normal">
                (the model never finished — its thinking ran past the token
                budget)
              </span>
            )}
          </div>
        </div>
      </div>

      {/* INTERPRETATION badge */}
      <div
        className="border rounded-lg p-4 flex gap-3 items-start"
        style={{
          borderColor: toneColor,
          backgroundColor: `${toneColor}15`,
        }}
      >
        <div
          className="w-1.5 h-1.5 rounded-full mt-2 shrink-0"
          style={{ backgroundColor: toneColor }}
        />
        <div className="flex-1">
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-1">
            What this step shows
          </div>
          <div className="text-sm leading-relaxed">{interpretation.label}</div>
          {step.cot_evidence ? (
            <details className="mt-3">
              <summary className="text-xs text-[var(--ink-faint)] cursor-pointer">
                judge evidence
              </summary>
              <div className="text-xs italic text-[var(--ink-soft)] mt-2 pl-3 border-l border-[var(--border)]">
                {step.cot_evidence}
              </div>
            </details>
          ) : null}
        </div>
      </div>
    </div>
  );
}
