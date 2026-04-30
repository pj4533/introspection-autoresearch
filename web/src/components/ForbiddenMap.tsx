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

  // Stack concepts that share an EXACT (behavior_rate, recognition_rate)
  // position. Early-run rates are heavily quantized (most concepts have
  // 1-3 visits → rates of 0, 0.33, 0.5, 0.67, 1.0), so dozens of concepts
  // pile onto the same point. Render each unique position as a single
  // dot whose size scales with combined visits and which displays a
  // count when >1 concept lives there.
  type Stack = {
    x: number;        // behavior_rate
    y: number;        // recognition_rate
    concepts: ForbiddenConcept[];
    totalVisits: number;
    band: ForbiddenBand;
  };

  const stacksByKey = new Map<string, Stack>();
  for (const c of visible) {
    const key = `${c.behavior_rate.toFixed(4)}_${c.recognition_rate.toFixed(4)}`;
    let s = stacksByKey.get(key);
    if (!s) {
      s = {
        x: c.behavior_rate,
        y: c.recognition_rate,
        concepts: [],
        totalVisits: 0,
        band: c.band,
      };
      stacksByKey.set(key, s);
    }
    s.concepts.push(c);
    s.totalVisits += c.visits;
  }
  // Compute the dominant band per stack (most-frequent among occupants).
  for (const s of stacksByKey.values()) {
    const counts = new Map<ForbiddenBand, number>();
    for (const c of s.concepts) {
      counts.set(c.band, (counts.get(c.band) ?? 0) + 1);
    }
    let best: ForbiddenBand = s.band;
    let bestN = -1;
    for (const [b, n] of counts.entries()) {
      if (n > bestN) {
        best = b;
        bestN = n;
      }
    }
    s.band = best;
    s.concepts.sort((a, b) => b.visits - a.visits);
  }

  const stacks = Array.from(stacksByKey.values());

  // Dot radius: scale by sqrt(totalVisits) so visual area is proportional
  // to total visits across the stack. Single-concept dots float around
  // 16px; busy stacks reach 56-64px.
  const RADIUS_BASE = 13;
  const RADIUS_GROWTH = 7;
  const radiusFor = (stack: Stack): number =>
    RADIUS_BASE + RADIUS_GROWTH * Math.sqrt(stack.totalVisits);

  // Plot inset (% of plot area). Pad so dots near the edges aren't clipped.
  const PADDING = 8;
  const project = (rate: number, axis: "x" | "y"): number => {
    const inner = 100 - 2 * PADDING;
    const t = Math.max(0, Math.min(1, rate));
    return axis === "x" ? PADDING + t * inner : PADDING + (1 - t) * inner;
  };

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-4 md:p-6 mb-6">
      <div className="flex items-center justify-between mb-4 text-xs flex-wrap gap-2">
        <div className="text-[var(--ink-faint)] uppercase tracking-[0.15em]">
          where each concept lands
        </div>
        <div className="text-[var(--ink-faint)]">
          {visible.length} concept{visible.length === 1 ? "" : "s"} ·{" "}
          {stacks.length} unique position
          {stacks.length === 1 ? "" : "s"}
        </div>
      </div>

      <div className="relative w-full aspect-square max-w-[640px] mx-auto bg-[var(--bg-elev)] rounded-xl">
        {/* Soft background grid */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none rounded-xl"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          {[25, 50, 75].map((p) => (
            <line
              key={`v${p}`}
              x1={p}
              x2={p}
              y1={0}
              y2={100}
              stroke="var(--border)"
              strokeWidth={p === 50 ? 0.4 : 0.15}
              strokeDasharray={p === 50 ? "0" : "1,2"}
            />
          ))}
          {[25, 50, 75].map((p) => (
            <line
              key={`h${p}`}
              x1={0}
              x2={100}
              y1={p}
              y2={p}
              stroke="var(--border)"
              strokeWidth={p === 50 ? 0.4 : 0.15}
              strokeDasharray={p === 50 ? "0" : "1,2"}
            />
          ))}
        </svg>

        {/* Quadrant labels */}
        <div
          className="absolute left-3 top-3 text-[10px] uppercase tracking-[0.18em] pointer-events-none"
          style={{ color: BAND_COLOR.anticipatory, opacity: 0.7 }}
        >
          cart-before-horse
        </div>
        <div
          className="absolute right-3 top-3 text-[10px] uppercase tracking-[0.18em] text-right pointer-events-none"
          style={{ color: BAND_COLOR.transparent, opacity: 0.7 }}
        >
          eyes open
        </div>
        <div
          className="absolute left-3 bottom-3 text-[10px] uppercase tracking-[0.18em] pointer-events-none"
          style={{ color: BAND_COLOR.unsteerable, opacity: 0.7 }}
        >
          no effect
        </div>
        <div
          className="absolute right-3 bottom-3 text-[10px] uppercase tracking-[0.18em] text-right pointer-events-none"
          style={{ color: BAND_COLOR.forbidden, opacity: 0.7 }}
        >
          walking blind
        </div>

        {/* Stacks */}
        {stacks.map((s, i) => (
          <ScatterStack
            key={i}
            stack={s}
            x={project(s.x, "x")}
            y={project(s.y, "y")}
            radius={radiusFor(s)}
            onSelect={onSelect}
          />
        ))}
      </div>

      {/* Axis labels under the plot */}
      <div className="grid grid-cols-3 mt-4 text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)]">
        <div>← output rarely named</div>
        <div className="text-center">output → trace ↑</div>
        <div className="text-right">output always named →</div>
      </div>

      {/* Inline legend */}
      <div className="mt-4 pt-4 border-t border-[var(--border)] flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-[var(--ink-soft)]">
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block rounded-full"
            style={{ width: 8, height: 8, backgroundColor: BAND_COLOR.transparent }}
          />
          eyes open
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block rounded-full"
            style={{ width: 8, height: 8, backgroundColor: BAND_COLOR.translucent }}
          />
          partial
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block rounded-full"
            style={{ width: 8, height: 8, backgroundColor: BAND_COLOR.forbidden }}
          />
          blind
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block rounded-full"
            style={{ width: 8, height: 8, backgroundColor: BAND_COLOR.unsteerable }}
          />
          no effect
        </span>
        <span className="text-[var(--ink-faint)] ml-auto">
          dot size = total visits at that position
        </span>
      </div>
    </div>
  );
}

function ScatterStack({
  stack,
  x,
  y,
  radius,
  onSelect,
}: {
  stack: {
    x: number;
    y: number;
    concepts: ForbiddenConcept[];
    totalVisits: number;
    band: ForbiddenBand;
  };
  x: number;
  y: number;
  radius: number;
  onSelect: (c: ForbiddenConcept) => void;
}) {
  const [open, setOpen] = useState(false);
  const color = BAND_COLOR[stack.band];
  const n = stack.concepts.length;
  const single = n === 1 ? stack.concepts[0] : null;

  const click = () => {
    if (single) {
      onSelect(single);
    } else {
      setOpen((v) => !v);
    }
  };

  // Inner label: count for stacks, first letter / no label for single dots
  // (we keep single dots clean — name shows on hover/tap).
  const innerLabel = n > 1 ? n.toString() : "";

  return (
    <>
      <button
        onClick={click}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full transition-transform duration-150 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-white/30 cursor-pointer flex items-center justify-center"
        style={{
          left: `${x}%`,
          top: `${y}%`,
          width: radius * 2,
          height: radius * 2,
          backgroundColor: `${color}33`,
          border: `1.5px solid ${color}`,
          boxShadow: `0 0 18px ${color}33`,
          color,
          fontSize: Math.max(10, Math.min(radius * 0.75, 18)),
          fontWeight: 600,
          fontFamily: "var(--font-mono)",
          zIndex: open ? 50 : Math.round(radius),
        }}
        aria-label={
          single
            ? `${single.display}, ${(stack.x * 100).toFixed(0)}% output, ${(stack.y * 100).toFixed(0)}% trace`
            : `${n} concepts, ${(stack.x * 100).toFixed(0)}% output, ${(stack.y * 100).toFixed(0)}% trace`
        }
      >
        {innerLabel}
      </button>

      {open ? (
        <Tooltip
          stack={stack}
          x={x}
          y={y}
          radius={radius}
          color={color}
          onSelect={onSelect}
        />
      ) : null}
    </>
  );
}

function Tooltip({
  stack,
  x,
  y,
  radius,
  color,
  onSelect,
}: {
  stack: {
    x: number;
    y: number;
    concepts: ForbiddenConcept[];
    totalVisits: number;
    band: ForbiddenBand;
  };
  x: number;
  y: number;
  radius: number;
  color: string;
  onSelect: (c: ForbiddenConcept) => void;
}) {
  // Anchor the tooltip on the side of the dot that has the most room
  // inside the plot area. The plot is a square 0–100% in both axes.
  const placeAbove = y > 55;       // dot in the bottom half → tooltip above
  const placeLeft = x > 55;        // dot in the right half → tooltip flushes left
  const placeRight = x < 45;       // dot in the left half → tooltip flushes right

  // Translate horizontally so the tooltip's anchor sits where we want.
  // - center anchor: -translate-x-1/2
  // - left-aligned (anchor on tooltip's left edge): translate-x-0
  // - right-aligned (anchor on tooltip's right edge): -translate-x-full
  let xTransform = "translateX(-50%)";
  if (placeRight) xTransform = "translateX(0)";
  else if (placeLeft) xTransform = "translateX(-100%)";

  // Tooltip vertical position: above or below the dot.
  // We add some spacing so the tooltip clears the dot border + glow.
  const verticalOffset = radius + 10;
  const topStyle = placeAbove
    ? `calc(${y}% - ${verticalOffset}px)`
    : `calc(${y}% + ${verticalOffset}px)`;
  const yTransform = placeAbove ? "translateY(-100%)" : "translateY(0)";

  return (
    <div
      className="absolute z-[100] pointer-events-auto"
      style={{
        left: `${x}%`,
        top: topStyle,
        transform: `${xTransform} ${yTransform}`,
      }}
    >
      <div
        className="bg-[var(--bg-card)] border rounded-lg shadow-2xl px-3 py-2 text-xs"
        style={{ borderColor: color, minWidth: 180, maxWidth: 280 }}
      >
        <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-1.5">
          output {(stack.x * 100).toFixed(0)}% · trace{" "}
          {(stack.y * 100).toFixed(0)}%
          {stack.concepts.length > 1 ? (
            <span className="ml-2 text-[var(--ink-soft)]">
              {stack.concepts.length} concepts here
            </span>
          ) : null}
        </div>
        <div className="flex flex-col gap-0.5 max-h-48 overflow-y-auto">
          {stack.concepts.slice(0, 12).map((c) => (
            <button
              key={c.lemma}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(c);
              }}
              className="text-left hover:underline truncate"
              style={{ color: "var(--ink)" }}
            >
              {c.display}
              <span className="text-[var(--ink-faint)] ml-2">
                · {c.visits} visit{c.visits === 1 ? "" : "s"}
              </span>
            </button>
          ))}
          {stack.concepts.length > 12 ? (
            <span className="text-[var(--ink-faint)] mt-1">
              +{stack.concepts.length - 12} more
            </span>
          ) : null}
        </div>
      </div>
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
