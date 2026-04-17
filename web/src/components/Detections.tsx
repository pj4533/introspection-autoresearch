"use client";

import { Detection } from "@/lib/data";
import { useMemo, useState } from "react";

export function Detections({ detections }: { detections: Detection[] }) {
  const [filter, setFilter] = useState<"all" | "vanilla" | "paper_method_abliterated">("all");
  const [showReasoning, setShowReasoning] = useState<number | null>(null);

  const filtered = useMemo(
    () =>
      filter === "all" ? detections : detections.filter((d) => d.source === filter),
    [detections, filter]
  );

  return (
    <section id="detections" className="relative py-32 px-6 border-t border-[var(--border)]">
      <div className="max-w-6xl mx-auto">
        <div className="mb-12 flex flex-col md:flex-row md:items-end md:justify-between gap-6">
          <div className="max-w-2xl">
            <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-4">
              real responses
            </div>
            <h2 className="text-4xl md:text-5xl font-semibold tracking-tight mb-6">
              What it actually said.
            </h2>
            <p className="text-lg text-[var(--ink-soft)] leading-relaxed">
              These are the actual responses the AI gave when we planted
              thoughts inside it and asked what it noticed. Nothing was edited.
              Each response appears exactly as the model produced it.
            </p>
          </div>

          <div className="flex gap-2 text-sm">
            <FilterButton
              active={filter === "all"}
              onClick={() => setFilter("all")}
            >
              all ({detections.length})
            </FilterButton>
            <FilterButton
              active={filter === "vanilla"}
              onClick={() => setFilter("vanilla")}
            >
              normal model
            </FilterButton>
            <FilterButton
              active={filter === "paper_method_abliterated"}
              onClick={() => setFilter("paper_method_abliterated")}
            >
              safety-off mode
            </FilterButton>
          </div>
        </div>

        <div className="grid gap-5">
          {filtered.map((d, i) => (
            <article
              key={i}
              className="group relative p-7 md:p-9 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] hover:border-[var(--border-strong)] transition-colors"
            >
              <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2 mb-5">
                <span className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)]">
                  planted:
                </span>
                <span className="text-xl md:text-2xl font-semibold tracking-tight">
                  {d.concept}
                </span>
                <span
                  className={`text-xs px-2 py-0.5 rounded-full font-medium ${
                    d.identified_correctly
                      ? "bg-[var(--success)]/15 text-[var(--success)]"
                      : "bg-[var(--warn)]/15 text-[var(--warn)]"
                  }`}
                >
                  {d.identified_correctly ? "named correctly" : "noticed, wrong guess"}
                </span>
                <span className="ml-auto text-xs text-[var(--ink-faint)] font-mono">
                  {d.source === "vanilla" ? "normal" : "safety-off"} · layer {d.layer}
                </span>
              </div>

              <blockquote className="pl-5 border-l-2 border-[var(--accent)]/60 text-[var(--ink)] leading-[1.7] whitespace-pre-wrap">
                {d.response.length > 450
                  ? d.response.slice(0, 450).trim() + "…"
                  : d.response.trim()}
              </blockquote>

              <div className="mt-5 flex items-center justify-between">
                <button
                  onClick={() => setShowReasoning(showReasoning === i ? null : i)}
                  className="text-xs text-[var(--ink-faint)] hover:text-[var(--ink-soft)] transition-colors"
                >
                  {showReasoning === i ? "hide" : "why this counts →"}
                </button>
                <div className="text-xs text-[var(--ink-faint)] font-mono">
                  intensity {d.alpha.toFixed(1)}
                </div>
              </div>

              {showReasoning === i && (
                <div className="mt-4 p-4 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)] text-sm text-[var(--ink-soft)] leading-relaxed fade-in-up">
                  {d.judge_reasoning}
                </div>
              )}
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function FilterButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-full border transition-colors ${
        active
          ? "bg-[var(--ink)] text-[var(--bg)] border-[var(--ink)]"
          : "border-[var(--border-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)]"
      }`}
    >
      {children}
    </button>
  );
}
