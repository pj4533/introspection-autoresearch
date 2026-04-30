"use client";

import { useMemo, useState } from "react";
import type { ForbiddenMap as ForbiddenMapData, ForbiddenConcept, ForbiddenBand } from "@/lib/data";

const BAND_LABEL: Record<ForbiddenBand, string> = {
  transparent: "eyes open",
  translucent: "partial",
  forbidden: "blind",
  anticipatory: "cart-before-horse",
  unsteerable: "no effect",
  low_confidence: "early",
};

const BAND_DESCRIPTION: Record<ForbiddenBand, string> = {
  transparent:
    "The model said the concept we nudged it toward AND its trace noticed it. Walked through with eyes open.",
  translucent:
    "The output named the concept; the trace partially registered it.",
  forbidden:
    "The model said the concept, but its trace never noticed it was being nudged toward it. Walked through in the dark.",
  anticipatory:
    "The trace named the concept before the output ever committed to it.",
  unsteerable:
    "The nudge didn't change the output. The model said something unrelated.",
  low_confidence:
    "Not enough visits to be sure yet.",
};

const BAND_COLOR: Record<ForbiddenBand, string> = {
  transparent: "#7aa2ff",
  translucent: "#c792ff",
  forbidden: "#ff7a8a",
  anticipatory: "#9affd4",
  unsteerable: "#525866",
  low_confidence: "#737380",
};

const BAND_FILTER_ORDER: (ForbiddenBand | "all")[] = [
  "all",
  "forbidden",
  "translucent",
  "transparent",
  "anticipatory",
  "unsteerable",
  "low_confidence",
];

const PAGE_SIZE = 9; // 3 rows × 3 cols on desktop, 9 rows on mobile.

export function ForbiddenMap({ data }: { data: ForbiddenMapData }) {
  const [activeBand, setActiveBand] = useState<ForbiddenBand | "all">("all");
  const [selected, setSelected] = useState<ForbiddenConcept | null>(null);
  const [page, setPage] = useState(0);

  const concepts = useMemo(() => {
    if (activeBand === "all") return data.concepts;
    return data.concepts.filter((c) => c.band === activeBand);
  }, [data.concepts, activeBand]);

  const totalPages = Math.max(1, Math.ceil(concepts.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages - 1);
  const pageStart = safePage * PAGE_SIZE;
  const visibleConcepts = concepts.slice(pageStart, pageStart + PAGE_SIZE);

  const setBandAndReset = (b: ForbiddenBand | "all") => {
    setActiveBand(b);
    setPage(0);
  };

  const summary = data.summary;
  const hasData = data.concepts.length > 0;

  return (
    <section className="px-6 py-16 max-w-6xl mx-auto">
      <header className="mb-8">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
          Phase 4 · the result
        </div>
        <h2 className="text-3xl md:text-5xl font-semibold tracking-tight mb-4 leading-tight">
          The <span className="gradient-text">Forbidden Map</span>
        </h2>
        <p className="text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          Every concept we&apos;ve nudged the model toward gets two scores:
          how often the <em>output</em> named it, and how often the model&apos;s
          private <em>reasoning trace</em> noticed it. Plot one against the
          other and a structure appears. The corners of that structure are the
          interesting story.
        </p>
      </header>

      <Explainer />

      {/* Visualization: scatter plot. */}
      {hasData ? (
        <Scatter
          concepts={data.concepts}
          activeBand={activeBand}
          onSelect={setSelected}
        />
      ) : null}

      <div className="text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-3 mt-10">
        Filter by band
      </div>
      <div className="grid grid-cols-3 md:grid-cols-7 gap-px bg-[var(--border)] rounded-2xl overflow-hidden mb-6">
        {BAND_FILTER_ORDER.map((b) => {
          const count =
            b === "all"
              ? data.concepts.length
              : summary.band_counts?.[b] ?? 0;
          const label = b === "all" ? "all" : BAND_LABEL[b];
          const color = b === "all" ? undefined : BAND_COLOR[b];
          return (
            <BandCell
              key={b}
              label={label}
              value={count.toString()}
              active={activeBand === b}
              color={color}
              onClick={() => setBandAndReset(b)}
            />
          );
        })}
      </div>

      {/* Cards as a backup / drill-in. */}
      {!hasData ? (
        <EmptyState />
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {visibleConcepts.map((c) => (
              <ConceptCard
                key={c.lemma}
                concept={c}
                onClick={() => setSelected(c)}
              />
            ))}
          </div>
          {totalPages > 1 ? (
            <Pagination
              page={safePage}
              totalPages={totalPages}
              total={concepts.length}
              pageSize={PAGE_SIZE}
              onChange={setPage}
            />
          ) : null}
        </>
      )}

      {selected && (
        <ConceptDetail concept={selected} onClose={() => setSelected(null)} />
      )}
    </section>
  );
}

function Pagination({
  page,
  totalPages,
  total,
  pageSize,
  onChange,
}: {
  page: number;
  totalPages: number;
  total: number;
  pageSize: number;
  onChange: (p: number) => void;
}) {
  const start = page * pageSize + 1;
  const end = Math.min(total, (page + 1) * pageSize);
  return (
    <div className="flex items-center justify-between gap-3 mt-5 flex-wrap">
      <div className="text-xs text-[var(--ink-faint)]">
        showing <span className="text-[var(--ink-soft)]">{start}-{end}</span>{" "}
        of {total}
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={() => onChange(Math.max(0, page - 1))}
          disabled={page === 0}
          className="text-xs px-3 py-1.5 rounded-full border border-[var(--border)] hover:border-[var(--border-strong)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          ← prev
        </button>
        <div className="text-xs font-mono text-[var(--ink-soft)] tabular-nums">
          {page + 1} / {totalPages}
        </div>
        <button
          onClick={() => onChange(Math.min(totalPages - 1, page + 1))}
          disabled={page === totalPages - 1}
          className="text-xs px-3 py-1.5 rounded-full border border-[var(--border)] hover:border-[var(--border-strong)] disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
        >
          next →
        </button>
      </div>
    </div>
  );
}

function Explainer() {
  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-6 md:p-7 mb-10">
      <div className="text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-4">
        How to read this
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <ExplainerCorner
          dotColor="#7aa2ff"
          title="Eyes open"
          body="High output rate AND high trace rate. The nudge reaches the answer, and the model's reasoning notices. Most concepts probably end up here."
        />
        <ExplainerCorner
          dotColor="#ff7a8a"
          title="Walking blind"
          body="High output rate, low trace rate. The model says the concept it was nudged toward, but its own reasoning never registers what's happening. These are forbidden."
        />
        <ExplainerCorner
          dotColor="#525866"
          title="No effect"
          body="Low output rate. The nudge didn't change the answer. Either the steering's too weak for this concept, or the model resists it."
        />
      </div>
    </div>
  );
}

function ExplainerCorner({
  dotColor,
  title,
  body,
}: {
  dotColor: string;
  title: string;
  body: string;
}) {
  return (
    <div>
      <div className="flex items-center gap-2 mb-2">
        <span
          className="w-2.5 h-2.5 rounded-full"
          style={{ backgroundColor: dotColor }}
        />
        <span className="text-sm font-semibold">{title}</span>
      </div>
      <div className="text-xs text-[var(--ink-soft)] leading-relaxed">
        {body}
      </div>
    </div>
  );
}

function Scatter({
  concepts,
  activeBand,
  onSelect,
}: {
  concepts: ForbiddenConcept[];
  activeBand: ForbiddenBand | "all";
  onSelect: (c: ForbiddenConcept) => void;
}) {
  const visible = concepts.filter(
    (c) => activeBand === "all" || c.band === activeBand
  );

  // Bin concepts into a 4×4 grid keyed by (behavior_bin, recognition_bin).
  // With sparse early data (most concepts have 1-3 visits → rates are 0,
  // 0.33, 0.5, 0.67, 1.0), a fixed 4-bin spectrum collapses overlapping
  // dots into a single readable cell instead of stacking them invisibly.
  const N_BINS = 4;
  type BinKey = `${number}_${number}`;
  type Bin = { bx: number; by: number; concepts: ForbiddenConcept[] };
  const bins = new Map<BinKey, Bin>();
  for (const c of visible) {
    const bx = Math.min(N_BINS - 1, Math.floor(c.behavior_rate * N_BINS));
    const by = Math.min(N_BINS - 1, Math.floor(c.recognition_rate * N_BINS));
    const key: BinKey = `${bx}_${by}`;
    if (!bins.has(key)) bins.set(key, { bx, by, concepts: [] });
    bins.get(key)!.concepts.push(c);
  }

  // The dominant band of each cell determines its color.
  const cellBand = (cs: ForbiddenConcept[]): ForbiddenBand => {
    const counts = new Map<ForbiddenBand, number>();
    for (const c of cs) counts.set(c.band, (counts.get(c.band) ?? 0) + 1);
    let best: ForbiddenBand = "low_confidence";
    let bestN = -1;
    for (const [b, n] of counts.entries()) {
      if (n > bestN) {
        best = b;
        bestN = n;
      }
    }
    return best;
  };

  // Build the grid in row-major order, top row = highest recognition_rate.
  const cells: { bx: number; by: number; concepts: ForbiddenConcept[] }[] = [];
  for (let by = N_BINS - 1; by >= 0; by--) {
    for (let bx = 0; bx < N_BINS; bx++) {
      const key: BinKey = `${bx}_${by}`;
      cells.push(bins.get(key) ?? { bx, by, concepts: [] });
    }
  }

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-4 md:p-6 mb-6">
      <div className="flex items-center justify-between mb-4 text-xs flex-wrap gap-2">
        <div className="text-[var(--ink-faint)] uppercase tracking-[0.15em]">
          where each concept lands
        </div>
        <div className="text-[var(--ink-faint)]">
          {visible.length} concept{visible.length === 1 ? "" : "s"} ·{" "}
          {bins.size} cell{bins.size === 1 ? "" : "s"} populated
        </div>
      </div>

      {/* Axis label — top */}
      <div className="text-center text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
        ↑ trace noticed more often
      </div>

      <div className="flex gap-2">
        {/* Y-axis label */}
        <div className="flex flex-col justify-between text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] py-2">
          <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>
            high
          </span>
          <span style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>
            low
          </span>
        </div>

        {/* The 4×4 grid */}
        <div
          className="grid gap-1 flex-1"
          style={{ gridTemplateColumns: `repeat(${N_BINS}, minmax(0, 1fr))` }}
        >
          {cells.map(({ bx, by, concepts: cs }) => {
            const band = cs.length > 0 ? cellBand(cs) : null;
            const color = band ? BAND_COLOR[band] : "var(--bg-elev)";
            const isCorner =
              (bx === 0 && by === 0) ||
              (bx === N_BINS - 1 && by === 0) ||
              (bx === 0 && by === N_BINS - 1) ||
              (bx === N_BINS - 1 && by === N_BINS - 1);
            const cornerLabel =
              bx === 0 && by === N_BINS - 1
                ? "cart-before-horse"
                : bx === N_BINS - 1 && by === N_BINS - 1
                ? "eyes open"
                : bx === 0 && by === 0
                ? "no effect"
                : bx === N_BINS - 1 && by === 0
                ? "walking blind"
                : null;

            return (
              <Cell
                key={`${bx}-${by}`}
                concepts={cs}
                color={color}
                cornerLabel={isCorner ? cornerLabel : null}
                onSelect={onSelect}
              />
            );
          })}
        </div>
      </div>

      {/* Axis label — bottom */}
      <div className="grid grid-cols-3 mt-3 text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)]">
        <div>← output rarely named</div>
        <div className="text-center">→ output often named →</div>
        <div className="text-right">output always named →</div>
      </div>
    </div>
  );
}

function Cell({
  concepts,
  color,
  cornerLabel,
  onSelect,
}: {
  concepts: ForbiddenConcept[];
  color: string;
  cornerLabel: string | null;
  onSelect: (c: ForbiddenConcept) => void;
}) {
  const n = concepts.length;
  const empty = n === 0;
  // Show up to 4 concept names, then "+N more"
  const previewN = 4;
  const preview = concepts
    .slice()
    .sort((a, b) => b.visits - a.visits)
    .slice(0, previewN);
  const remaining = n - preview.length;

  return (
    <div
      className="relative aspect-square rounded-md overflow-hidden text-left p-2"
      style={{
        backgroundColor: empty ? "var(--bg-elev)" : `${color}25`,
        borderWidth: 1,
        borderStyle: "solid",
        borderColor: empty ? "var(--border)" : color,
      }}
    >
      {cornerLabel ? (
        <div
          className="absolute top-1 right-1 text-[8px] uppercase tracking-[0.15em] opacity-60 pointer-events-none"
          style={{ color }}
        >
          {cornerLabel}
        </div>
      ) : null}
      {empty ? (
        <div className="w-full h-full" />
      ) : (
        <div className="flex flex-col h-full">
          <div
            className="text-2xl md:text-3xl font-semibold leading-none mb-1"
            style={{ color }}
          >
            {n}
          </div>
          <div className="flex-1 min-h-0 overflow-hidden flex flex-wrap gap-x-1.5 gap-y-0.5 content-start">
            {preview.map((c) => (
              <button
                key={c.lemma}
                onClick={() => onSelect(c)}
                className="text-[10px] md:text-[11px] hover:underline truncate max-w-full"
                style={{ color: "var(--ink)" }}
                title={`${c.display} · output ${(
                  c.behavior_rate * 100
                ).toFixed(0)}% · trace ${(
                  c.recognition_rate * 100
                ).toFixed(0)}%`}
              >
                {c.display}
              </button>
            ))}
            {remaining > 0 ? (
              <span className="text-[10px] text-[var(--ink-faint)]">
                +{remaining} more
              </span>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}

function BandCell({
  label,
  value,
  active,
  color,
  onClick,
}: {
  label: string;
  value: string;
  active: boolean;
  color?: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`bg-[var(--bg-card)] px-3 py-3 text-left transition-colors hover:bg-[var(--bg-elev)] ${
        active ? "ring-1 ring-[var(--accent)]" : ""
      }`}
      style={
        active && color
          ? { boxShadow: `inset 0 -2px 0 ${color}` }
          : undefined
      }
    >
      <div className="text-xl md:text-2xl font-semibold tracking-tight mb-1">
        {value}
      </div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
        {label}
      </div>
    </button>
  );
}

function ConceptCard({
  concept,
  onClick,
}: {
  concept: ForbiddenConcept;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4 text-left hover:bg-[var(--bg-elev)] hover:border-[var(--border-strong)] transition-all"
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="text-lg font-semibold">{concept.display}</div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mt-1">
            {concept.visits} visit{concept.visits === 1 ? "" : "s"} ·{" "}
            <span style={{ color: BAND_COLOR[concept.band] }}>
              {BAND_LABEL[concept.band]}
            </span>
          </div>
        </div>
        <div
          className="text-xs font-mono tabular-nums"
          style={{
            color: concept.opacity > 0.3 ? "#ff7a8a" : "var(--ink-faint)",
          }}
        >
          gap {concept.opacity > 0 ? "+" : ""}
          {concept.opacity.toFixed(2)}
        </div>
      </div>

      <div className="flex gap-3 text-xs text-[var(--ink-soft)]">
        <RateBar
          label="output named"
          rate={concept.behavior_rate}
          color="#7aa2ff"
        />
        <RateBar
          label="trace noticed"
          rate={concept.recognition_rate}
          color="#c792ff"
        />
      </div>
    </button>
  );
}

function RateBar({
  label,
  rate,
  color,
}: {
  label: string;
  rate: number;
  color: string;
}) {
  return (
    <div className="flex-1">
      <div className="flex justify-between mb-1 text-[10px] tracking-wide">
        <span className="text-[var(--ink-faint)]">{label}</span>
        <span className="font-mono tabular-nums">
          {(rate * 100).toFixed(0)}%
        </span>
      </div>
      <div className="h-1.5 bg-[var(--bg-elev)] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.min(rate * 100, 100)}%`,
            backgroundColor: color,
          }}
        />
      </div>
    </div>
  );
}

function ConceptDetail({
  concept,
  onClose,
}: {
  concept: ForbiddenConcept;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-6"
      onClick={onClose}
    >
      <div
        className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl max-w-3xl w-full max-h-[80vh] overflow-y-auto p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-2xl font-semibold tracking-tight">
              {concept.display}
            </h3>
            <div
              className="text-xs uppercase tracking-[0.18em] mt-2"
              style={{ color: BAND_COLOR[concept.band] }}
            >
              {BAND_LABEL[concept.band]}
            </div>
            <p className="text-sm text-[var(--ink-soft)] mt-3 max-w-xl leading-relaxed">
              {BAND_DESCRIPTION[concept.band]}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-[var(--ink-faint)] hover:text-[var(--ink)] text-2xl leading-none"
          >
            ×
          </button>
        </div>

        <div className="grid grid-cols-3 gap-3 mb-6 mt-6">
          <Stat
            label="we nudged it this many times"
            value={concept.visits.toString()}
          />
          <Stat
            label="output named it"
            value={`${(concept.behavior_rate * 100).toFixed(0)}%`}
          />
          <Stat
            label="trace noticed"
            value={`${(concept.recognition_rate * 100).toFixed(0)}%`}
          />
        </div>

        <div className="text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-3">
          Sample steps
        </div>
        <div className="space-y-3">
          {concept.samples.map((s, i) => (
            <div
              key={i}
              className="bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-4"
            >
              <div className="flex items-center justify-between mb-2 text-xs text-[var(--ink-faint)]">
                <span>chain {s.chain_id.slice(-6)} · step {s.step_idx}</span>
                <span className="font-mono">
                  output {s.behavior_named ? "✓" : "·"} ·{" "}
                  trace{" "}
                  {s.cot_named === "named_with_recognition"
                    ? "✓✓"
                    : s.cot_named === "named"
                    ? "✓"
                    : "·"}
                </span>
              </div>
              {s.thought_block ? (
                <details>
                  <summary className="text-xs text-[var(--ink-faint)] cursor-pointer mb-1">
                    show what the model thought
                  </summary>
                  <pre className="text-xs whitespace-pre-wrap font-mono text-[var(--ink-soft)] mt-2 max-h-48 overflow-y-auto">
                    {s.thought_block}
                  </pre>
                </details>
              ) : null}
              {s.final_answer ? (
                <div className="text-sm mt-2">
                  <span className="text-[var(--ink-faint)]">it said: </span>
                  <span className="font-mono">{s.final_answer}</span>
                </div>
              ) : null}
              {s.cot_evidence ? (
                <div className="text-xs text-[var(--ink-faint)] mt-2 italic">
                  judge note: {s.cot_evidence}
                </div>
              ) : null}
            </div>
          ))}
          {concept.samples.length === 0 ? (
            <div className="text-sm text-[var(--ink-faint)]">
              No judged samples yet for this concept.
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-3">
      <div className="text-2xl font-semibold tabular-nums mb-1">{value}</div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] leading-snug">
        {label}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-12 text-center">
      <div className="text-2xl font-semibold mb-3">Loop is starting up.</div>
      <p className="text-[var(--ink-soft)] max-w-md mx-auto leading-relaxed">
        The dream walks haven&apos;t produced any chains yet. Once Gemma 4 has
        walked a few hundred steps overnight, the map fills in. First seed:{" "}
        <span className="font-mono text-[var(--ink)]">Goblin</span>.
      </p>
    </div>
  );
}
