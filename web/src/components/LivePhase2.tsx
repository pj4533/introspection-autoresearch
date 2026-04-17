"use client";

import { Phase2Entry, Phase2Activity, Summary } from "@/lib/data";
import { timeAgo } from "@/lib/utils";
import { useEffect, useState } from "react";

export function LivePhase2({
  leaderboard,
  activity,
  summary,
}: {
  leaderboard: Phase2Entry[];
  activity: Phase2Activity[];
  summary: Summary;
}) {
  const [ago, setAgo] = useState(timeAgo(summary.last_updated));
  useEffect(() => {
    const id = setInterval(() => setAgo(timeAgo(summary.last_updated)), 30000);
    return () => clearInterval(id);
  }, [summary.last_updated]);

  const topPhase2 = leaderboard.slice(0, 12);

  return (
    <section id="live" className="relative py-32 px-6 border-t border-[var(--border)]">
      <div className="max-w-6xl mx-auto">
        <div className="mb-12 flex flex-col md:flex-row md:items-end md:justify-between gap-6">
          <div className="max-w-2xl">
            <div className="flex items-center gap-3 mb-4">
              <span className="w-2 h-2 rounded-full bg-[var(--success)] pulse-dot" />
              <span className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)]">
                running right now
              </span>
            </div>
            <h2 className="text-4xl md:text-5xl font-semibold tracking-tight mb-6">
              The machine is hunting for new thoughts to plant.
            </h2>
            <p className="text-lg text-[var(--ink-soft)] leading-relaxed">
              Overnight, a second program is generating thousands of candidate
              &ldquo;thoughts&rdquo; — including ones that don&apos;t have a
              single-word name, like <em>grounded-assertion</em> or{" "}
              <em>inhabiting-vs-reporting</em> — and testing whether the AI
              can notice them. The scoreboard below is the current top of the
              pile, updated automatically.
            </p>
          </div>
          <div className="text-right text-xs text-[var(--ink-faint)]">
            <div>last refresh: {ago}</div>
            <div className="mt-1">data rebuilds ~every 15 min</div>
          </div>
        </div>

        <div className="grid md:grid-cols-3 gap-4 mb-10">
          <StatCard
            label="candidates tested"
            value={summary.phase2_candidates_evaluated.toString()}
          />
          <StatCard
            label="candidates that worked"
            value={summary.phase2_candidates_with_hits.toString()}
            accent
          />
          <StatCard
            label="top score"
            value={summary.phase2_top_score.toFixed(3)}
          />
        </div>

        {topPhase2.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] overflow-hidden">
            <div className="grid grid-cols-12 gap-4 px-6 py-3 text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] border-b border-[var(--border)]">
              <div className="col-span-1">#</div>
              <div className="col-span-5 md:col-span-4">thought</div>
              <div className="col-span-2 hidden md:block">kind</div>
              <div className="col-span-3 md:col-span-2 text-right">score</div>
              <div className="col-span-3 text-right">noticed / total</div>
            </div>
            {topPhase2.map((row, i) => (
              <LeaderRow key={row.candidate_id} row={row} rank={i + 1} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function StatCard({
  label,
  value,
  accent = false,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="p-6 rounded-xl bg-[var(--bg-card)] border border-[var(--border)]">
      <div
        className={`text-3xl font-semibold tracking-tight mb-1 ${
          accent ? "text-[var(--accent)]" : ""
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

function LeaderRow({ row, rank }: { row: Phase2Entry; rank: number }) {
  const isContrast = row.derivation_method === "contrast_pair";
  const name = isContrast && row.contrast_pair ? row.contrast_pair.axis : row.concept;
  const kind = isContrast ? "invented" : "dictionary word";

  return (
    <div className="grid grid-cols-12 gap-4 px-6 py-4 border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--bg-elev)] transition-colors">
      <div className="col-span-1 text-sm text-[var(--ink-faint)] font-mono">
        {String(rank).padStart(2, "0")}
      </div>
      <div className="col-span-5 md:col-span-4 text-sm">
        <div className="font-medium">{name}</div>
        {isContrast && row.contrast_pair?.description && (
          <div className="text-xs text-[var(--ink-faint)] mt-0.5 truncate">
            {row.contrast_pair.description}
          </div>
        )}
      </div>
      <div className="col-span-2 hidden md:block text-xs text-[var(--ink-soft)]">
        {kind}
      </div>
      <div
        className={`col-span-3 md:col-span-2 text-right font-mono text-sm ${
          row.score > 0.3
            ? "text-[var(--accent)]"
            : row.score > 0.05
            ? "text-[var(--ink)]"
            : "text-[var(--ink-faint)]"
        }`}
      >
        {row.score.toFixed(3)}
      </div>
      <div className="col-span-3 text-right font-mono text-sm text-[var(--ink-soft)]">
        {(row.detection_rate * 100).toFixed(0)}% <span className="text-[var(--ink-faint)]">at layer {row.layer}</span>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="p-16 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] border-dashed text-center">
      <div className="text-[var(--ink-soft)] mb-2">
        The scoreboard is empty right now.
      </div>
      <div className="text-xs text-[var(--ink-faint)]">
        Overnight run starts tonight. Check back in the morning.
      </div>
    </div>
  );
}
