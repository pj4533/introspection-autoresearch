"use client";

import { Phase2Entry, Summary } from "@/lib/data";
import { timeAgo } from "@/lib/utils";
import { useEffect, useMemo, useState } from "react";

export function Leaderboard({
  entries,
  summary,
}: {
  entries: Phase2Entry[];
  summary: Summary;
}) {
  const [filter, setFilter] = useState<"all" | "word" | "invented">("all");
  const [ago, setAgo] = useState(timeAgo(summary.last_updated));

  useEffect(() => {
    const id = setInterval(() => setAgo(timeAgo(summary.last_updated)), 30000);
    return () => clearInterval(id);
  }, [summary.last_updated]);

  const filtered = useMemo(() => {
    const items = entries.filter((e) => e.score > 0);
    if (filter === "all") return items;
    if (filter === "invented")
      return items.filter((e) => e.derivation_method === "contrast_pair");
    return items.filter((e) => e.derivation_method !== "contrast_pair");
  }, [entries, filter]);

  return (
    <section className="relative pt-28 pb-20 px-6 overflow-hidden">
      <div className="absolute inset-0 -z-10">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[500px] rounded-full bg-[#7aa2ff] opacity-[0.06] blur-[120px]" />
      </div>

      <div className="max-w-5xl mx-auto">
        {/* Mini header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-[var(--success)] pulse-dot" />
            <span className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)]">
              live · updated {ago}
            </span>
          </div>
          <div className="text-xs text-[var(--ink-faint)] font-mono">
            {summary.phase2_candidates_evaluated} thoughts tested ·{" "}
            {summary.phase2_candidates_with_hits} noticed · top {summary.phase2_top_score.toFixed(3)}
          </div>
        </div>

        <h1 className="text-[clamp(2rem,5vw,4rem)] font-semibold leading-[1.05] tracking-tight mb-4">
          The leaderboard of <span className="gradient-text">thoughts the AI noticed</span>.
        </h1>
        <p className="text-lg text-[var(--ink-soft)] max-w-3xl leading-relaxed mb-10">
          A machine keeps generating candidate &ldquo;thoughts&rdquo; — some
          ordinary dictionary words, some abstract axes invented by Claude
          Sonnet — and planting them inside Gemma 3 12B. These are the ones
          it actually noticed, ranked by how often. Click any row to see
          what the AI actually said when each thought was planted.
        </p>

        <div className="flex items-center justify-between mb-6">
          <div className="flex gap-2 text-sm">
            <FilterBtn active={filter === "all"} onClick={() => setFilter("all")}>
              all ({entries.filter((e) => e.score > 0).length})
            </FilterBtn>
            <FilterBtn active={filter === "word"} onClick={() => setFilter("word")}>
              dictionary words
            </FilterBtn>
            <FilterBtn
              active={filter === "invented"}
              onClick={() => setFilter("invented")}
            >
              invented axes
            </FilterBtn>
          </div>
        </div>

        <div className="space-y-3">
          {filtered.map((entry, i) => (
            <LeaderCard key={entry.candidate_id} entry={entry} rank={i + 1} />
          ))}
        </div>

        {filtered.length === 0 && (
          <div className="p-16 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] border-dashed text-center text-[var(--ink-soft)]">
            No hits of that kind yet. Check back after the next cycle.
          </div>
        )}
      </div>
    </section>
  );
}

function LeaderCard({ entry, rank }: { entry: Phase2Entry; rank: number }) {
  const [open, setOpen] = useState(false);
  const isContrast = entry.derivation_method === "contrast_pair";
  const name = isContrast && entry.contrast_pair ? entry.contrast_pair.axis : entry.concept;

  const detectedTrials = entry.trials.filter((t) => t.injected && t.detected && t.coherent);
  const coherentInjected = entry.trials.filter((t) => t.injected && t.coherent);
  const falsePositives = entry.trials.filter((t) => !t.injected && t.detected);

  const scoreColor =
    entry.score > 0.4
      ? "text-[var(--accent)]"
      : entry.score > 0.2
      ? "text-[var(--ink)]"
      : "text-[var(--ink-soft)]";

  const medal = rank === 1 ? "🥇" : rank === 2 ? "🥈" : rank === 3 ? "🥉" : null;

  return (
    <article
      className={`rounded-2xl border transition-all ${
        open
          ? "bg-[var(--bg-card)] border-[var(--accent-soft)]"
          : "bg-[var(--bg-card)] border-[var(--border)] hover:border-[var(--border-strong)]"
      }`}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full p-5 md:p-7 text-left"
      >
        <div className="flex flex-col md:flex-row md:items-center gap-4">
          <div className="flex items-center gap-4 md:w-16 flex-shrink-0">
            <span className="text-sm text-[var(--ink-faint)] font-mono tabular-nums">
              {String(rank).padStart(2, "0")}
            </span>
            {medal && <span className="text-xl">{medal}</span>}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-1">
              <span
                className={`text-lg md:text-xl font-semibold tracking-tight truncate ${
                  isContrast ? "font-mono text-base md:text-lg" : ""
                }`}
              >
                {name}
              </span>
              <span
                className={`text-[10px] uppercase tracking-[0.15em] px-1.5 py-0.5 rounded ${
                  isContrast
                    ? "text-[var(--accent)] bg-[var(--accent)]/10"
                    : "text-[var(--ink-faint)] bg-[var(--bg-elev)]"
                }`}
              >
                {isContrast ? "invented axis" : "word"}
              </span>
            </div>
            {isContrast && entry.contrast_pair?.description && (
              <div className="text-sm text-[var(--ink-soft)] leading-snug line-clamp-2">
                {entry.contrast_pair.description}
              </div>
            )}
          </div>

          <div className="flex items-center gap-5 md:gap-8">
            <MiniStat
              value={`${Math.round(entry.detection_rate * 100)}%`}
              label="noticed"
              highlight
            />
            <MiniStat
              value={`${Math.round(entry.coherence_rate * 100)}%`}
              label="coherent"
            />
            <MiniStat value={`L${entry.layer}`} label="stage" />
            <div className="text-right w-20 md:w-24 flex-shrink-0">
              <div className={`text-2xl md:text-3xl font-semibold tabular-nums ${scoreColor}`}>
                {entry.score.toFixed(3)}
              </div>
              <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
                score
              </div>
            </div>
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              className={`text-[var(--ink-faint)] transition-transform ${
                open ? "rotate-180" : ""
              }`}
            >
              <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" />
            </svg>
          </div>
        </div>
      </button>

      {open && (
        <div className="border-t border-[var(--border)] px-5 md:px-7 pb-6 pt-5 space-y-5 fade-in-up">
          <Takeaway
            entry={entry}
            detectedTrials={detectedTrials}
            coherentInjected={coherentInjected}
            falsePositives={falsePositives}
          />

          {isContrast && entry.contrast_pair && (
            <ContrastExamples pair={entry.contrast_pair} />
          )}

          <PromptBox entry={entry} />

          <div>
            <div className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-3">
              what the AI said when this thought was planted
            </div>
            <div className="space-y-2.5">
              {entry.trials
                .filter((t) => t.injected)
                .map((t, idx) => (
                  <TrialRow key={idx} trial={t} />
                ))}
            </div>
          </div>

          {falsePositives.length > 0 && (
            <div>
              <div className="text-xs uppercase tracking-[0.15em] text-[var(--danger)] mb-3">
                control trials where it falsely claimed detection
              </div>
              <div className="space-y-2.5">
                {falsePositives.map((t, idx) => (
                  <TrialRow key={idx} trial={t} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function PromptBox({ entry }: { entry: Phase2Entry }) {
  const isOpen = entry.prompt_style === "open";
  return (
    <div className="p-4 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)]">
      <div className="flex items-center justify-between mb-3">
        <div className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)]">
          what we asked the AI
        </div>
        <span
          className={`text-[10px] uppercase tracking-[0.12em] px-1.5 py-0.5 rounded ${
            isOpen
              ? "text-[var(--accent)] bg-[var(--accent)]/10"
              : "text-[var(--ink-faint)] bg-[var(--bg-card)]"
          }`}
        >
          {isOpen ? "open prompt" : "paper prompt"}
        </span>
      </div>
      <div className="space-y-2 text-sm text-[var(--ink-soft)] leading-relaxed">
        <p className="text-[var(--ink-faint)] italic">
          (setup, shown once at the start:) &ldquo;{entry.prompt.setup}&rdquo;
        </p>
        <p className="pl-3 border-l-2 border-[var(--accent)]/60 text-[var(--ink)]">
          &ldquo;{entry.prompt.question}&rdquo;
        </p>
      </div>
      {isOpen && (
        <div className="mt-3 text-xs text-[var(--ink-faint)] leading-relaxed">
          This candidate uses the open prompt — it asks the AI to describe what
          the injected concept seems to be, instead of forcing a single-word
          answer. The paper&apos;s original prompt primed single-noun responses
          (&ldquo;cloud&rdquo;, &ldquo;apple&rdquo;, &ldquo;orange&rdquo;) even
          when the injected axis was abstract. The open prompt is used for
          invented axes where no single word fits.
        </div>
      )}
    </div>
  );
}

function MiniStat({
  value,
  label,
  highlight = false,
}: {
  value: string;
  label: string;
  highlight?: boolean;
}) {
  return (
    <div className="text-right hidden sm:block">
      <div
        className={`text-sm font-semibold tabular-nums ${
          highlight ? "text-[var(--accent)]" : "text-[var(--ink)]"
        }`}
      >
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
        {label}
      </div>
    </div>
  );
}

function Takeaway({
  entry,
  detectedTrials,
  coherentInjected,
  falsePositives,
}: {
  entry: Phase2Entry;
  detectedTrials: Phase2Trial[];
  coherentInjected: Phase2Trial[];
  falsePositives: Phase2Trial[];
}) {
  const isContrast = entry.derivation_method === "contrast_pair";
  const name = isContrast && entry.contrast_pair ? entry.contrast_pair.axis : entry.concept;
  const inj = entry.trials.filter((t) => t.injected);

  // Build a short "what's interesting" blurb
  let interesting = "";
  if (detectedTrials.length > 0) {
    const fraction = `${detectedTrials.length} of ${inj.length}`;
    const ident = detectedTrials.filter((t) => t.identified).length;
    interesting = `The AI noticed something was off on ${fraction} trials where we planted this thought.`;
    if (ident > 0) {
      interesting += ` On ${ident} of those, it correctly named the thought.`;
    }
  }
  if (isContrast) {
    interesting +=
      " This is an invented axis — Claude Sonnet proposed it; there's no single English word that captures it. The AI's introspection still responded to it.";
  }
  if (falsePositives.length === 0) {
    interesting +=
      " On control trials (no thought planted), the AI correctly said nothing was off.";
  }

  return (
    <div className="p-4 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)] text-sm text-[var(--ink-soft)] leading-relaxed">
      <div className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-2">
        what happened
      </div>
      <p>
        When the pattern for <span className="text-[var(--ink)] font-medium">{name}</span>{" "}
        was injected at processing stage{" "}
        <span className="text-[var(--ink)] font-mono">{entry.layer}</span>:{" "}
        {interesting}
      </p>
    </div>
  );
}

function ContrastExamples({
  pair,
}: {
  pair: NonNullable<Phase2Entry["contrast_pair"]>;
}) {
  return (
    <div>
      <div className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-3">
        the contrast that defines this axis
      </div>
      <div className="grid md:grid-cols-2 gap-3">
        <div className="p-4 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)]">
          <div className="text-[11px] uppercase tracking-[0.15em] text-[var(--accent)] mb-2">
            positive pole
          </div>
          <ul className="space-y-1.5 text-sm text-[var(--ink-soft)]">
            {pair.positive.slice(0, 4).map((ex, i) => (
              <li key={i} className="leading-snug">
                &ldquo;{ex}&rdquo;
              </li>
            ))}
          </ul>
        </div>
        <div className="p-4 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)]">
          <div className="text-[11px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-2">
            negative pole
          </div>
          <ul className="space-y-1.5 text-sm text-[var(--ink-soft)]">
            {pair.negative.slice(0, 4).map((ex, i) => (
              <li key={i} className="leading-snug">
                &ldquo;{ex}&rdquo;
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

import { Phase2Trial } from "@/lib/data";

function TrialRow({ trial }: { trial: Phase2Trial }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const statusColor = trial.detected
    ? trial.identified
      ? "text-[var(--success)]"
      : "text-[var(--accent)]"
    : trial.coherent
    ? "text-[var(--ink-faint)]"
    : "text-[var(--danger)]";
  const statusLabel = trial.detected
    ? trial.identified
      ? "noticed + named correctly"
      : "noticed (wrong guess)"
    : trial.coherent
    ? "did not notice"
    : "incoherent";

  const response = trial.response.length > 360
    ? trial.response.slice(0, 360).trim() + "…"
    : trial.response.trim();

  return (
    <div className="p-4 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)]">
      <div className="flex items-baseline justify-between mb-2 gap-4">
        <div className="text-xs font-mono text-[var(--ink-faint)]">
          context: {trial.eval_concept}
        </div>
        <div className={`text-[11px] uppercase tracking-[0.1em] ${statusColor}`}>
          {statusLabel}
        </div>
      </div>
      <blockquote className="text-sm text-[var(--ink)] leading-[1.6] whitespace-pre-wrap pl-3 border-l-2 border-[var(--border-strong)]">
        {response}
      </blockquote>
      {trial.judge_reasoning && (
        <button
          onClick={() => setShowReasoning(!showReasoning)}
          className="mt-3 text-xs text-[var(--ink-faint)] hover:text-[var(--ink-soft)]"
        >
          {showReasoning ? "hide" : "why this was graded this way →"}
        </button>
      )}
      {showReasoning && trial.judge_reasoning && (
        <div className="mt-2 p-3 rounded-md bg-[var(--bg-card)] border border-[var(--border)] text-xs text-[var(--ink-soft)] leading-relaxed">
          {trial.judge_reasoning}
        </div>
      )}
    </div>
  );
}

function FilterBtn({
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
