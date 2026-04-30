"use client";

import { useEffect, useState } from "react";
import { timeAgo } from "@/lib/utils";
import type { ForbiddenMapSummary } from "@/lib/data";

export function Phase4Hero({ summary }: { summary: ForbiddenMapSummary }) {
  const [ago, setAgo] = useState<string>(timeAgo(summary.last_updated));
  useEffect(() => {
    if (!summary.last_updated) return;
    const id = setInterval(
      () => setAgo(timeAgo(summary.last_updated)),
      30000
    );
    return () => clearInterval(id);
  }, [summary.last_updated]);

  return (
    <section className="relative pt-40 pb-24 px-6 overflow-hidden">
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[600px] rounded-full bg-[#ff7a8a] opacity-[0.06] blur-[120px]" />
        <div className="absolute top-40 left-[10%] w-[500px] h-[500px] rounded-full bg-[#c792ff] opacity-[0.05] blur-[120px]" />
      </div>

      <div className="max-w-4xl mx-auto text-center fade-in-up">
        <div className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)] pulse-dot" />
          phase 4 · gemma 4 · live
          {summary.last_updated ? <> · updated {ago}</> : null}
        </div>

        <h1 className="text-[clamp(2.75rem,7vw,5rem)] font-semibold leading-[1.05] tracking-tight mb-6 max-w-3xl mx-auto">
          Thoughts an AI can have{" "}
          <span className="gradient-text">but can&apos;t notice.</span>
        </h1>

        <p className="text-lg md:text-xl text-[var(--ink-soft)] max-w-2xl mx-auto leading-relaxed mb-10">
          We secretly nudge an open-source AI toward a concept and watch two
          things: what word comes out, and whether its private reasoning
          notices the nudge. Sometimes both. Sometimes only one. The gap
          between them is the map below.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[var(--border)] rounded-2xl overflow-hidden max-w-3xl mx-auto mb-2">
          <StatCell
            label="dream walks so far"
            value={summary.n_chains.toLocaleString()}
            sub="each walk is up to 20 steps"
          />
          <StatCell
            label="nudges measured"
            value={summary.total_steps.toLocaleString()}
            sub="one per step across all walks"
            accent
          />
          <StatCell
            label="concepts seen"
            value={summary.n_concepts.toLocaleString()}
            sub="seeds plus self-generated"
          />
          <StatCell
            label="forbidden so far"
            value={(summary.band_counts?.forbidden ?? 0).toString()}
            sub="output bent, trace blind"
            danger
          />
        </div>

        <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mt-4 mb-2">
          how walks ended
        </div>
        <div className="grid grid-cols-3 gap-px bg-[var(--border)] rounded-2xl overflow-hidden max-w-3xl mx-auto">
          <EndReasonCell
            label="looped back"
            value={summary.n_self_loop.toLocaleString()}
            sub="walked into a concept already visited"
            color="#7aa2ff"
          />
          <EndReasonCell
            label="trailed off"
            value={summary.n_coherence_break.toLocaleString()}
            sub="answer unparseable / thinking ran long"
            color="#c792ff"
          />
          <EndReasonCell
            label="full 20 steps"
            value={summary.n_length_cap.toLocaleString()}
            sub="rare — most stop sooner"
            color="#9affd4"
          />
        </div>
      </div>
    </section>
  );
}

function EndReasonCell({
  label,
  value,
  sub,
  color,
}: {
  label: string;
  value: string;
  sub: string;
  color: string;
}) {
  return (
    <div className="bg-[var(--bg-card)] px-4 py-4 md:px-5 md:py-5">
      <div
        className="text-2xl md:text-3xl font-semibold tracking-tight mb-1 tabular-nums"
        style={{ color }}
      >
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
        {label}
      </div>
      <div className="text-[10px] text-[var(--ink-faint)] mt-1.5 leading-snug">
        {sub}
      </div>
    </div>
  );
}

function StatCell({
  label,
  value,
  sub,
  accent = false,
  danger = false,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: boolean;
  danger?: boolean;
}) {
  return (
    <div className="bg-[var(--bg-card)] px-4 py-5 md:px-6 md:py-6">
      <div
        className={`text-3xl md:text-4xl font-semibold tracking-tight mb-1 ${
          accent
            ? "text-[var(--accent)]"
            : danger
            ? "text-[#ff7a8a]"
            : "text-[var(--ink)]"
        }`}
      >
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
        {label}
      </div>
      {sub ? (
        <div className="text-[10px] text-[var(--ink-faint)] mt-1.5 leading-snug">
          {sub}
        </div>
      ) : null}
    </div>
  );
}
