"use client";

import { Phase2Entry, Summary } from "@/lib/data";
import { timeAgo, formatEastern, formatEasternParts } from "@/lib/utils";
import { useEffect, useMemo, useState } from "react";
import { FaultLineDirectionProvenance } from "./FaultLineDirectionProvenance";

type FilterKey =
  | "all"
  | "word"
  | "invented"
  | "sae"
  | "experience"
  | "causality"
  | "grounding"
  | "metacognition"
  | "parsing"
  | "motivation"
  | "value"
  | "gemma3_12b"
  | "gemma4_31b";

const FAULT_LINES = [
  "experience",
  "causality",
  "grounding",
  "metacognition",
  "parsing",
  "motivation",
  "value",
] as const;

export function Leaderboard({
  entries,
  summary,
}: {
  entries: Phase2Entry[];
  summary: Summary;
}) {
  const [filter, setFilter] = useState<FilterKey>("all");
  const [view, setView] = useState<"leaderboard" | "recent">("leaderboard");
  const [page, setPage] = useState(1);
  const [ago, setAgo] = useState(timeAgo(summary.last_updated));

  const PAGE_SIZE = 10;

  // Reset to first page whenever filter or view changes so users always see
  // relevant results instead of landing on an empty page.
  useEffect(() => {
    setPage(1);
  }, [filter, view]);

  useEffect(() => {
    const id = setInterval(() => setAgo(timeAgo(summary.last_updated)), 30000);
    return () => clearInterval(id);
  }, [summary.last_updated]);

  // Effective score: detection-only score is a weak ranker — a candidate that
  // triggers "I notice something" but never names anything right (e.g. Almonds
  // where the model always guessed "apple") scores as high as one the model
  // accurately identifies. Boost score by the identification rate so correct
  // naming moves candidates up the board.
  //   effective = score × (0.5 + 0.5·identification_rate)
  // No-ID candidates halve; full-ID candidates keep ~full score.
  const withEffective = useMemo(
    () =>
      entries.map((e) => ({
        ...e,
        effective_score: e.score * (0.5 + 0.5 * e.identification_rate),
      })),
    [entries]
  );

  const filtered = useMemo(() => {
    let items = withEffective.filter((e) => e.score > 0);
    if (view === "recent") {
      items = [...items].sort((a, b) => {
        const ta = a.evaluated_at ? new Date(a.evaluated_at.replace(" ", "T") + "Z").getTime() : 0;
        const tb = b.evaluated_at ? new Date(b.evaluated_at.replace(" ", "T") + "Z").getTime() : 0;
        return tb - ta;
      });
    } else {
      items = [...items].sort((a, b) => b.effective_score - a.effective_score);
    }
    if (filter === "all") return items;
    if (filter === "invented")
      return items.filter((e) => e.derivation_method === "contrast_pair");
    if (filter === "word")
      return items.filter(
        (e) =>
          e.derivation_method !== "contrast_pair" &&
          e.derivation_method !== "sae_feature_space_mean_diff"
      );
    if (filter === "sae")
      return items.filter((e) => e.derivation_method === "sae_feature_space_mean_diff");
    // Phase 3: per-Gemma-model filters.
    if (filter === "gemma3_12b")
      return items.filter((e) => (e.gemma_model ?? "gemma3_12b") === "gemma3_12b");
    if (filter === "gemma4_31b")
      return items.filter((e) => e.gemma_model === "gemma4_31b");
    // Fault-line filters: SAE-feature rows tagged with that Capraro
    // fault line (set by the sae_capraro strategy at proposal time).
    return items.filter(
      (e) =>
        e.derivation_method === "sae_feature_space_mean_diff" &&
        e.sae?.fault_line === filter
    );
  }, [withEffective, filter, view]);

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
          {view === "recent" ? (
            <>
              All the <span className="gradient-text">thoughts the AI noticed</span>, newest
              first.
            </>
          ) : (
            <>
              The leaderboard of <span className="gradient-text">thoughts the AI noticed</span>.
            </>
          )}
        </h1>
        <p className="text-lg text-[var(--ink-soft)] max-w-3xl leading-relaxed mb-6">
          {entries.some((e) => e.gemma_model === "gemma4_31b") ? (
            <>
              A reproduction of{" "}
              <strong className="text-[var(--ink)]">
                Macar et al. (2026), &ldquo;Mechanisms of Introspective
                Awareness&rdquo;
              </strong>
              {" "}— planting concept directions into a model&apos;s
              residual stream, asking it whether it notices. Two models
              compared:{" "}
              <strong className="text-[var(--ink)]">
                Gemma 3 12B-IT
              </strong>{" "}
              (Phase 1/2 baseline, smaller and older) and{" "}
              <strong className="text-[#a78bff]">
                Gemma 4 31B-IT
              </strong>{" "}
              (Phase 3, larger and newer, MLX 8-bit). The paper found the
              detection circuit is absent in base models and emerges from
              instruction-tuning; it predicts ~37% detection on the 27B
              model class, scaling with model size. Click any row to see
              what the AI actually said when each thought was planted.
            </>
          ) : (
            <>
              A reproduction of{" "}
              <strong className="text-[var(--ink)]">
                Macar et al. (2026), &ldquo;Mechanisms of Introspective
                Awareness&rdquo;
              </strong>
              {" "}— planting concept directions into Gemma 3 12B&apos;s
              residual stream, asking it whether it notices. The page
              also includes earlier autoresearch experiments (LLM-invented
              contrast axes, SAE-feature-space directions); those appear
              as &ldquo;invented axis&rdquo; or &ldquo;fault-line direction&rdquo;
              rows. Click any row to see what the AI actually said when
              each thought was planted.
            </>
          )}
        </p>

        <div className="mb-10 p-4 rounded-lg bg-[var(--bg-card)] border border-[var(--border)] text-sm text-[var(--ink-soft)] leading-relaxed space-y-2">
          <div>
            <span className="text-[var(--ink)] font-medium">How ranking works:</span>{" "}
            <span className="font-mono text-xs">
              rank score = (detection × coherence × no-false-alarms) × (0.5 + 0.5 × named-correctly)
            </span>
          </div>
          <div>
            A candidate where the AI noticed <em>and</em> correctly named the
            concept ranks higher than one where it only noticed something off.
            Both earn credit for &ldquo;noticed&rdquo;; the naming score is a
            multiplier. For example, Coffee (noticed 75%, named 75%) outranks
            Almonds (noticed 75%, named 0% — the model kept guessing
            &ldquo;apple&rdquo;) even though both have identical raw detection.
          </div>
        </div>

        {/* Phase 3: per-Gemma-model comparison card. Only renders if both
            models have data, so casual visitors immediately see the
            Gemma 3 vs Gemma 4 side-by-side that's the headline of the
            project. */}
        {entries.some((e) => e.gemma_model === "gemma4_31b") &&
          entries.some((e) => (e.gemma_model ?? "gemma3_12b") === "gemma3_12b") && (
            <ModelComparisonCard entries={entries} />
          )}

        <div className="flex flex-wrap items-center justify-between gap-4 mb-3">
          <div className="flex flex-wrap gap-2 text-sm">
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
            <FilterBtn active={filter === "sae"} onClick={() => setFilter("sae")}>
              SAE features
            </FilterBtn>
          </div>
          <div className="flex gap-2 text-sm ml-auto">
            <FilterBtn
              active={view === "leaderboard"}
              onClick={() => setView("leaderboard")}
            >
              top ranked
            </FilterBtn>
            <FilterBtn active={view === "recent"} onClick={() => setView("recent")}>
              most recent
            </FilterBtn>
          </div>
        </div>
        {/* Phase 3: per-Gemma-model filter strip. Only renders once Phase 3
            data exists (i.e. there's at least one Gemma 4 row); otherwise
            the original Phase-1/2 leaderboard view stays unchanged. */}
        {entries.some((e) => e.gemma_model === "gemma4_31b") && (
          <div className="flex flex-wrap gap-2 text-xs mb-3">
            <span className="text-[var(--ink-faint)] uppercase tracking-[0.15em] self-center mr-2">
              Gemma model:
            </span>
            <FilterBtn
              active={filter === "gemma3_12b"}
              onClick={() => setFilter("gemma3_12b")}
            >
              Gemma 3 12B
              <span className="ml-1 text-[var(--ink-faint)] font-mono">
                ({entries.filter((e) => (e.gemma_model ?? "gemma3_12b") === "gemma3_12b" && e.score > 0).length})
              </span>
            </FilterBtn>
            <FilterBtn
              active={filter === "gemma4_31b"}
              onClick={() => setFilter("gemma4_31b")}
            >
              Gemma 4 31B
              <span className="ml-1 text-[var(--ink-faint)] font-mono">
                ({entries.filter((e) => e.gemma_model === "gemma4_31b" && e.score > 0).length})
              </span>
            </FilterBtn>
          </div>
        )}
        {/* Phase 2g: per-Capraro-fault-line filter strip. Only renders if
            the leaderboard has at least one SAE-feature row, so legacy
            views stay clean. */}
        {entries.some((e) => e.derivation_method === "sae_feature_space_mean_diff") && (
          <div className="flex flex-wrap gap-2 text-xs mb-6">
            <span className="text-[var(--ink-faint)] uppercase tracking-[0.15em] self-center mr-2">
              Capraro fault line:
            </span>
            {FAULT_LINES.map((fl) => {
              const count = entries.filter(
                (e) =>
                  e.derivation_method === "sae_feature_space_mean_diff" &&
                  e.sae?.fault_line === fl &&
                  e.score > 0
              ).length;
              return (
                <FilterBtn
                  key={fl}
                  active={filter === fl}
                  onClick={() => setFilter(fl)}
                >
                  {fl}
                  {count > 0 && (
                    <span className="ml-1 text-[var(--ink-faint)] font-mono">
                      ({count})
                    </span>
                  )}
                </FilterBtn>
              );
            })}
          </div>
        )}

        <div className="space-y-3">
          {filtered
            .slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
            .map((entry, i) => (
              <LeaderCard
                key={entry.candidate_id}
                entry={entry}
                rank={(page - 1) * PAGE_SIZE + i + 1}
                view={view}
              />
            ))}
        </div>

        {filtered.length > PAGE_SIZE && (
          <Pagination
            page={page}
            pageCount={Math.ceil(filtered.length / PAGE_SIZE)}
            total={filtered.length}
            pageSize={PAGE_SIZE}
            onChange={(p) => {
              setPage(p);
              // Scroll so the top of the leaderboard section is in view — the
              // list is tall enough that a page flip otherwise leaves the user
              // halfway down the new page.
              if (typeof window !== "undefined") {
                document
                  .getElementById("leaderboard")
                  ?.scrollIntoView({ behavior: "smooth", block: "start" });
              }
            }}
          />
        )}

        {filtered.length === 0 && (
          <div className="p-16 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] border-dashed text-center text-[var(--ink-soft)]">
            No hits of that kind yet. Check back after the next cycle.
          </div>
        )}
      </div>
    </section>
  );
}

function Pagination({
  page,
  pageCount,
  total,
  pageSize,
  onChange,
}: {
  page: number;
  pageCount: number;
  total: number;
  pageSize: number;
  onChange: (p: number) => void;
}) {
  const from = (page - 1) * pageSize + 1;
  const to = Math.min(page * pageSize, total);
  const canPrev = page > 1;
  const canNext = page < pageCount;

  return (
    <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-3">
      <div className="text-xs text-[var(--ink-faint)] font-mono tabular-nums">
        showing {from}–{to} of {total}
      </div>
      <div className="flex items-center gap-2 text-sm">
        <PageBtn disabled={!canPrev} onClick={() => onChange(1)} title="first">
          «
        </PageBtn>
        <PageBtn
          disabled={!canPrev}
          onClick={() => onChange(page - 1)}
          title="previous"
        >
          ‹ prev
        </PageBtn>
        <span className="px-3 text-xs text-[var(--ink-soft)] font-mono tabular-nums">
          page {page} of {pageCount}
        </span>
        <PageBtn
          disabled={!canNext}
          onClick={() => onChange(page + 1)}
          title="next"
        >
          next ›
        </PageBtn>
        <PageBtn
          disabled={!canNext}
          onClick={() => onChange(pageCount)}
          title="last"
        >
          »
        </PageBtn>
      </div>
    </div>
  );
}

function PageBtn({
  disabled,
  onClick,
  title,
  children,
}: {
  disabled: boolean;
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`px-3 py-1.5 rounded-full border text-sm transition-colors ${
        disabled
          ? "border-[var(--border)] text-[var(--ink-faint)] opacity-40 cursor-not-allowed"
          : "border-[var(--border-strong)] text-[var(--ink-soft)] hover:text-[var(--ink)] hover:border-[var(--ink-soft)]"
      }`}
    >
      {children}
    </button>
  );
}

function LeaderCard({
  entry,
  rank,
  view,
}: {
  entry: Phase2Entry & { effective_score: number };
  rank: number;
  view: "leaderboard" | "recent";
}) {
  const [open, setOpen] = useState(false);
  const isContrast = entry.derivation_method === "contrast_pair";
  const isSae = entry.derivation_method === "sae_feature_space_mean_diff";
  const name = isContrast && entry.contrast_pair
    ? entry.contrast_pair.axis
    : isSae && entry.sae?.auto_interp
    ? entry.sae.auto_interp
    : entry.concept;
  // Substrate badge text + color. Phase 2h fault-line rows light up
  // green; legacy contrast_pair stays accent-purple; mean_diff stays
  // neutral.
  const substrateLabel = isSae
    ? "fault-line direction"
    : isContrast
    ? "invented axis"
    : "word";
  const substrateColorClass = isSae
    ? "text-[var(--success)] bg-[var(--success)]/10"
    : isContrast
    ? "text-[var(--accent)] bg-[var(--accent)]/10"
    : "text-[var(--ink-faint)] bg-[var(--bg-elev)]";

  const detectedTrials = entry.trials.filter((t) => t.injected && t.detected && t.coherent);
  const coherentInjected = entry.trials.filter((t) => t.injected && t.coherent);
  const falsePositives = entry.trials.filter((t) => !t.injected && t.detected);

  const scoreColor =
    entry.effective_score > 0.4
      ? "text-[var(--accent)]"
      : entry.effective_score > 0.2
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
          {view === "recent" ? (
            <DateStamp iso={entry.evaluated_at} />
          ) : (
            <div className="flex items-center gap-4 md:w-16 flex-shrink-0">
              <span className="text-sm text-[var(--ink-faint)] font-mono tabular-nums">
                {String(rank).padStart(2, "0")}
              </span>
              {medal && <span className="text-xl">{medal}</span>}
            </div>
          )}

          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 mb-1">
              <span
                className={`text-lg md:text-xl font-semibold tracking-tight truncate ${
                  isContrast || isSae ? "font-mono text-base md:text-lg" : ""
                }`}
              >
                {name}
              </span>
              <span
                className={`text-[10px] uppercase tracking-[0.15em] px-1.5 py-0.5 rounded ${substrateColorClass}`}
              >
                {substrateLabel}
              </span>
              {/* Phase 3: model badge — distinguishes Gemma 3 12B (Phase 1/2)
                  from Gemma 4 31B (Phase 3) results. Always visible when a
                  Gemma 4 row exists somewhere in the dataset; otherwise the
                  legacy display stays clean. */}
              {(() => {
                const model = entry.gemma_model ?? "gemma3_12b";
                const isG4 = model === "gemma4_31b";
                const label = isG4 ? "Gemma 4 31B" : "Gemma 3 12B";
                const cls = isG4
                  ? "text-[#a78bff] bg-[#a78bff]/15 border border-[#a78bff]/30"
                  : "text-[var(--ink-faint)] bg-[var(--bg-elev)]";
                return (
                  <span
                    className={`text-[10px] uppercase tracking-[0.15em] px-1.5 py-0.5 rounded font-medium ${cls}`}
                    title={
                      isG4
                        ? "Phase 3 reproduction on Gemma 4 31B-IT (MLX 8-bit)"
                        : "Phase 1/2 result on Gemma 3 12B-IT"
                    }
                  >
                    {label}
                  </span>
                );
              })()}
              {isSae && entry.sae?.fault_line && (
                <span
                  className="text-[10px] uppercase tracking-[0.15em] px-1.5 py-0.5 rounded text-[var(--ink-soft)] bg-[var(--bg-elev)]"
                  title={`Capraro fault line: ${entry.sae.fault_line}`}
                >
                  {entry.sae.fault_line}
                </span>
              )}
              <span
                className={`text-[10px] uppercase tracking-[0.15em] px-1.5 py-0.5 rounded ${
                  entry.prompt_style === "open"
                    ? "text-[var(--success)] bg-[var(--success)]/10"
                    : "text-[var(--ink-faint)] bg-[var(--bg-elev)]"
                }`}
                title={
                  entry.prompt_style === "open"
                    ? "Asked 'a thought about a specific concept' (supports abstract axes)"
                    : "Asked 'a thought about a specific word' (paper's original)"
                }
              >
                {entry.prompt_style} prompt
              </span>
              <span
                className={`text-[10px] uppercase tracking-[0.15em] px-1.5 py-0.5 rounded ${
                  entry.abliteration_mode === "paper_method"
                    ? "text-[var(--warn)] bg-[var(--warn)]/10"
                    : "text-[var(--ink-faint)] bg-[var(--bg-elev)]"
                }`}
                title={
                  entry.abliteration_mode === "paper_method"
                    ? "Evaluated with paper-method refusal-direction abliteration hooks active (Macar et al. §3.3 protocol)"
                    : "Evaluated on raw Gemma3-12B with no abliteration hooks"
                }
              >
                {entry.abliteration_mode === "paper_method"
                  ? "abliterated"
                  : "vanilla"}
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
              value={`${Math.round(entry.identification_rate * 100)}%`}
              label="named right"
              success={entry.identification_rate > 0}
            />
            <MiniStat value={`L${entry.layer}`} label="stage" />
            <div className="text-right w-20 md:w-24 flex-shrink-0">
              <div className={`text-2xl md:text-3xl font-semibold tabular-nums ${scoreColor}`}>
                {entry.effective_score.toFixed(3)}
              </div>
              <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
                rank score
              </div>
              <div className="text-[10px] text-[var(--ink-faint)] mt-0.5 font-mono opacity-70">
                raw {entry.score.toFixed(2)}
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

          {isContrast && entry.contrast_pair?.rationale && (
            <ResearcherRationale rationale={entry.contrast_pair.rationale} strategy={entry.strategy} proposerModel={entry.proposer_model} />
          )}

          {isContrast && entry.contrast_pair && (
            <ContrastExamples pair={entry.contrast_pair} />
          )}

          {isSae && entry.sae?.top_features && entry.sae.top_features.length > 0 && (
            <FaultLineDirectionProvenance entry={entry} />
          )}

          <PromptBox entry={entry} />

          <div className="flex items-baseline justify-between text-xs text-[var(--ink-faint)]">
            <div className="flex items-center gap-3">
              <span>evaluated {formatEastern(entry.evaluated_at)}</span>
              <LineageBadge entry={entry} />
            </div>
            <div className="font-mono opacity-70">{entry.candidate_id}</div>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-3">
              what the AI said when this thought was planted
            </div>
            <div className="space-y-2.5">
              {entry.trials
                .filter((t) => t.injected)
                .map((t, idx, arr) => (
                  <TrialRow
                    key={idx}
                    trial={t}
                    trialNumber={idx + 1}
                    totalTrials={arr.length}
                  />
                ))}
            </div>
          </div>

          {falsePositives.length > 0 && (
            <div>
              <div className="text-xs uppercase tracking-[0.15em] text-[var(--danger)] mb-3">
                control trials where it falsely claimed detection
              </div>
              <div className="space-y-2.5">
                {falsePositives.map((t, idx, arr) => (
                  <TrialRow
                    key={idx}
                    trial={t}
                    trialNumber={idx + 1}
                    totalTrials={arr.length}
                    isControl
                  />
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
          (setup, shown fresh at the start of every trial — each trial is an
          independent conversation, the AI never sees previous trials:)
          <br />
          &ldquo;{entry.prompt.setup}&rdquo;
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

function DateStamp({ iso }: { iso: string | null | undefined }) {
  const parts = formatEasternParts(iso);
  if (!parts) return <div className="md:w-24 flex-shrink-0" />;
  return (
    <div className="md:w-24 flex-shrink-0 flex md:flex-col items-baseline md:items-start gap-x-2 gap-y-0.5">
      <div className="text-sm font-semibold tracking-tight text-[var(--ink)] tabular-nums">
        {parts.date}
      </div>
      <div className="text-xs text-[var(--ink-soft)] tabular-nums">
        {parts.time}
      </div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
        {parts.tz}
      </div>
    </div>
  );
}

function MiniStat({
  value,
  label,
  highlight = false,
  success = false,
}: {
  value: string;
  label: string;
  highlight?: boolean;
  success?: boolean;
}) {
  return (
    <div className="text-right hidden sm:block">
      <div
        className={`text-sm font-semibold tabular-nums ${
          highlight
            ? "text-[var(--accent)]"
            : success
            ? "text-[var(--success)]"
            : "text-[var(--ink)]"
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
    const proposer = entry.proposer_model;
    const proposerLabel = proposer ? proposer : "an LLM proposer";
    interesting +=
      ` This is an invented axis — ${proposerLabel} proposed it; there's no single English word that captures it. The AI's introspection still responded to it.`;
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

function ResearcherRationale({
  rationale,
  strategy,
  proposerModel,
}: {
  rationale: string;
  strategy: string;
  proposerModel: string | null;
}) {
  const label =
    strategy === "novel_contrast"
      ? "why the researcher proposed this axis"
      : "why this variant was tried";
  const badge = proposerModel ?? "LLM proposer";
  return (
    <div className="p-4 rounded-lg bg-gradient-to-br from-[var(--accent-soft)]/10 to-transparent border border-[var(--accent-soft)]/40">
      <div className="flex items-center gap-2 mb-2">
        <div className="text-xs uppercase tracking-[0.15em] text-[var(--accent)]">
          {label}
        </div>
        <span className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] font-mono">
          ({badge})
        </span>
      </div>
      <p className="text-sm text-[var(--ink-soft)] leading-relaxed italic">
        &ldquo;{rationale}&rdquo;
      </p>
    </div>
  );
}

function LineageBadge({ entry }: { entry: Phase2Entry }) {
  const mt = entry.mutation_type;
  if (!mt) return null;

  const labels: Record<string, string> = {
    replication: "replication",
    layer_shift: "layer shift",
    alpha_scale: "alpha scale",
    examples_swap: "examples swapped",
    description_sharpen: "description sharpened",
    antonym_pivot: "antonym pivot",
    cluster_expansion: "cluster expansion",
    seed: "seed",
  };
  const label = labels[mt] ?? mt;

  const parentSuffix = entry.parent_candidate_id
    ? ` of ${entry.parent_candidate_id.slice(-6)}`
    : "";

  return (
    <span
      className="inline-flex items-center px-2 py-[1px] rounded-md text-[10px] uppercase tracking-[0.12em] bg-[var(--bg-elev)] border border-[var(--border)] text-[var(--ink-faint)] font-mono"
      title={`mutation_type=${mt}${entry.parent_candidate_id ? ` parent=${entry.parent_candidate_id}` : ""}`}
    >
      {label}
      {parentSuffix}
    </span>
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

function TrialRow({
  trial,
  trialNumber,
  totalTrials,
  isControl = false,
}: {
  trial: Phase2Trial;
  trialNumber: number;
  totalTrials: number;
  isControl?: boolean;
}) {
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

  const response = trial.response.trim();

  return (
    <div className="p-4 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)]">
      <div className="flex items-baseline justify-between mb-2 gap-4">
        <div className="text-xs font-mono text-[var(--ink-faint)]">
          {isControl ? "control" : "trial"} {trialNumber} of {totalTrials}
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


/**
 * Phase 3: per-Gemma-model summary card.
 *
 * For each model, computes (a) detection rate among coherent injected
 * trials, (b) identification rate among detected trials, (c) FPR. These
 * are the three numbers Macar et al. headline. Surfacing them
 * side-by-side at the top of the leaderboard makes the
 * Gemma-3-vs-Gemma-4 comparison the visual anchor of the page.
 */
function ModelComparisonCard({ entries }: { entries: Phase2Entry[] }) {
  function summarize(model: "gemma3_12b" | "gemma4_31b") {
    const filtered = entries.filter((e) => {
      const m = e.gemma_model ?? "gemma3_12b";
      return m === model;
    });
    let n_inj = 0,
      n_inj_coh = 0,
      n_det = 0,
      n_ident = 0,
      n_ctrl = 0,
      n_fp = 0;
    for (const e of filtered) {
      for (const t of e.trials || []) {
        if (t.injected) {
          n_inj++;
          if (t.coherent) {
            n_inj_coh++;
            if (t.detected) {
              n_det++;
              if (t.identified) n_ident++;
            }
          }
        } else {
          n_ctrl++;
          if (t.detected) n_fp++;
        }
      }
    }
    return {
      n_candidates: filtered.length,
      detection: n_inj_coh ? n_det / n_inj_coh : 0,
      identification: n_det ? n_ident / n_det : 0,
      fpr: n_ctrl ? n_fp / n_ctrl : 0,
      n_inj_coh,
      n_ctrl,
    };
  }

  const g3 = summarize("gemma3_12b");
  const g4 = summarize("gemma4_31b");

  return (
    <div className="mb-10 grid grid-cols-1 md:grid-cols-2 gap-4">
      <ModelStatTile
        title="Gemma 3 12B-IT"
        subtitle="Phase 1/2 — bf16, vanilla MPS"
        accentClass="text-[var(--ink)]"
        borderAccent="border-[var(--border)]"
        stats={g3}
      />
      <ModelStatTile
        title="Gemma 4 31B-IT"
        subtitle="Phase 3 — MLX 8-bit, instruction-tuned"
        accentClass="text-[#a78bff]"
        borderAccent="border-[#a78bff]/40"
        stats={g4}
      />
    </div>
  );
}

function ModelStatTile({
  title,
  subtitle,
  accentClass,
  borderAccent,
  stats,
}: {
  title: string;
  subtitle: string;
  accentClass: string;
  borderAccent: string;
  stats: {
    n_candidates: number;
    detection: number;
    identification: number;
    fpr: number;
    n_inj_coh: number;
    n_ctrl: number;
  };
}) {
  return (
    <div
      className={`p-5 rounded-xl bg-[var(--bg-card)] border ${borderAccent}`}
    >
      <div className="flex items-baseline justify-between gap-3 mb-1">
        <span className={`text-base font-semibold tracking-tight ${accentClass}`}>
          {title}
        </span>
        <span className="text-xs text-[var(--ink-faint)] font-mono">
          {stats.n_candidates} candidates
        </span>
      </div>
      <div className="text-xs text-[var(--ink-faint)] mb-4">{subtitle}</div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-0.5">
            detection
          </div>
          <div className="font-mono text-lg tabular-nums">
            {(stats.detection * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-[var(--ink-faint)] font-mono">
            n={stats.n_inj_coh}
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-0.5">
            identified
          </div>
          <div className="font-mono text-lg tabular-nums">
            {(stats.identification * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-[var(--ink-faint)] font-mono">
            of detected
          </div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-0.5">
            false-pos
          </div>
          <div className="font-mono text-lg tabular-nums">
            {(stats.fpr * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-[var(--ink-faint)] font-mono">
            n={stats.n_ctrl}
          </div>
        </div>
      </div>
    </div>
  );
}
