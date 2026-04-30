"use client";

import { useState } from "react";
import type { DreamWalksFile, DreamChain, DreamStep } from "@/lib/data";

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
      <header className="mb-6">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
          Phase 4 · Dream Walk Viewer
        </div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
          Watch a chain unfold
        </h2>
        <p className="text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          Each chain is a sequence of steered free-associations. We start by
          steering the model toward the seed concept, then take its answer and
          steer toward that. Step through to see the model walk through its own
          associative space.
        </p>
      </header>

      <div className="flex flex-wrap gap-2 mb-6">
        {data.chains.slice(0, 24).map((c, i) => (
          <button
            key={c.chain_id}
            onClick={() => {
              setSelectedIdx(i);
              setStepIdx(0);
            }}
            className={`text-xs px-3 py-1.5 rounded-full border transition-colors ${
              i === selectedIdx
                ? "bg-[var(--accent)] text-black border-[var(--accent)]"
                : "border-[var(--border)] hover:border-[var(--border-strong)] text-[var(--ink-soft)]"
            }`}
          >
            {c.seed_concept}
          </button>
        ))}
      </div>

      <ChainTimeline
        chain={chain}
        currentStep={safeStepIdx}
        onStepClick={setStepIdx}
      />

      {step ? <StepView step={step} /> : null}
    </section>
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
  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 mb-4">
      <div className="text-xs text-[var(--ink-faint)] mb-3">
        chain {chain.chain_id.slice(-8)} · seed{" "}
        <span className="text-[var(--ink)] font-mono">{chain.seed_concept}</span> ·
        ended {chain.end_reason}{" "}
        {chain.n_steps > 0 ? `(${chain.n_steps} steps)` : ""}
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
        ✓✓ named with recognition · ✓ named in trace · ✓? output only · · neither
      </div>
    </div>
  );
}

function StepView({ step }: { step: DreamStep }) {
  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-6">
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)]">
            steering toward
          </div>
          <div className="text-2xl font-semibold mt-1">{step.target_concept}</div>
        </div>
        <div className="text-xs font-mono text-[var(--ink-faint)] tabular-nums">
          α={step.alpha.toFixed(2)} · ‖dir‖={step.direction_norm.toFixed(1)}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
            thinking trace{" "}
            {step.cot_named === "named_with_recognition"
              ? "(named with recognition)"
              : step.cot_named === "named"
              ? "(named)"
              : "(none)"}
          </div>
          <pre className="text-xs whitespace-pre-wrap font-mono text-[var(--ink-soft)] bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-3 max-h-72 overflow-y-auto">
            {step.thought_block || "(empty)"}
          </pre>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
            final answer{" "}
            {step.behavior_named ? "(named the concept)" : "(did not name)"}
          </div>
          <div className="text-lg font-mono bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-3 min-h-16">
            {step.final_answer || "(empty)"}
          </div>
          {step.cot_evidence ? (
            <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mt-4 mb-2">
              evidence
            </div>
          ) : null}
          {step.cot_evidence ? (
            <div className="text-xs italic text-[var(--ink-soft)] bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-3">
              {step.cot_evidence}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
