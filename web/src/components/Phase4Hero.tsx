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

        <h1 className="text-[clamp(2.5rem,6vw,5rem)] font-semibold leading-[1.05] tracking-tight mb-6">
          Some thoughts the model can think
          <br className="hidden sm:block" />
          but{" "}
          <span className="gradient-text">cannot notice itself thinking.</span>
        </h1>

        <p className="text-lg md:text-xl text-[var(--ink-soft)] max-w-2xl mx-auto leading-relaxed mb-10">
          We let Gemma 4 free-associate. Then we steered each of its
          associations and let it free-associate again. Across thousands of
          steps overnight, two patterns emerged. Some concepts the model
          walked through with eyes open. Others it walked through in the
          dark.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[var(--border)] rounded-2xl overflow-hidden max-w-3xl mx-auto">
          <StatCell
            label="dream walks"
            value={summary.n_chains.toLocaleString()}
          />
          <StatCell
            label="steps measured"
            value={summary.total_steps.toLocaleString()}
            accent
          />
          <StatCell
            label="concepts mapped"
            value={summary.n_concepts.toLocaleString()}
          />
          <StatCell
            label="forbidden"
            value={(summary.band_counts?.forbidden ?? 0).toString()}
            danger
          />
        </div>
      </div>
    </section>
  );
}

function StatCell({
  label,
  value,
  accent = false,
  danger = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
  danger?: boolean;
}) {
  return (
    <div className="bg-[var(--bg-card)] px-6 py-6">
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
      <div className="text-[11px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
        {label}
      </div>
    </div>
  );
}
