"use client";

import { Summary } from "@/lib/data";
import { timeAgo } from "@/lib/utils";
import { useEffect, useState } from "react";

export function Hero({ summary }: { summary: Summary }) {
  const [ago, setAgo] = useState<string>(timeAgo(summary.last_updated));
  useEffect(() => {
    const id = setInterval(() => setAgo(timeAgo(summary.last_updated)), 30000);
    return () => clearInterval(id);
  }, [summary.last_updated]);

  return (
    <section className="relative pt-40 pb-28 px-6 overflow-hidden">
      {/* Background glow */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[600px] rounded-full bg-[#7aa2ff] opacity-[0.07] blur-[120px]" />
        <div className="absolute top-40 left-[10%] w-[500px] h-[500px] rounded-full bg-[#c792ff] opacity-[0.05] blur-[120px]" />
      </div>

      <div className="max-w-4xl mx-auto text-center fade-in-up">
        <div className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)] pulse-dot" />
          live · updated {ago}
        </div>

        <h1 className="text-[clamp(2.5rem,6vw,5rem)] font-semibold leading-[1.05] tracking-tight mb-6">
          We secretly plant a thought <br className="hidden sm:block" />
          inside an AI&apos;s mind. <span className="gradient-text">Does it notice?</span>
        </h1>

        <p className="text-lg md:text-xl text-[var(--ink-soft)] max-w-2xl mx-auto leading-relaxed mb-12">
          Researchers figured out how to reach inside a language model and tilt
          its thinking — without telling it. We&apos;re running that experiment
          live on {summary.model}, and watching what it says when we ask:{" "}
          <em>do you notice anything?</em>
        </p>

        {/* Stat strip */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-[var(--border)] rounded-2xl overflow-hidden max-w-3xl mx-auto">
          <StatCell label="thoughts planted" value={summary.total_trials.toLocaleString()} />
          <StatCell
            label="times it noticed"
            value={summary.total_detections.toString()}
            accent
          />
          <StatCell
            label="named it correctly"
            value={summary.total_identifications.toString()}
          />
          <StatCell
            label="false alarms"
            value={`${summary.vanilla_fpr === 0 && summary.abliterated_fpr === 0 ? "0" : ((summary.vanilla_fpr + summary.abliterated_fpr) / 2 * 100).toFixed(0) + "%"}`}
            subtle
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
  subtle = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
  subtle?: boolean;
}) {
  return (
    <div className="bg-[var(--bg-card)] px-6 py-6">
      <div
        className={`text-3xl md:text-4xl font-semibold tracking-tight mb-1 ${
          accent
            ? "text-[var(--accent)]"
            : subtle
            ? "text-[var(--success)]"
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
