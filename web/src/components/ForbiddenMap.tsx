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

      {/* Visualization: three exemplar cards. */}
      {hasData ? (
        <ThreeWaysHero
          concepts={data.concepts}
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
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
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
          dotColor="#9affd4"
          title="Cart-before-horse"
          body="Low output rate, high trace rate. The model's reasoning surfaced the concept, but its final answer committed to something else. The trace got there first."
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

function ThreeWaysHero({
  concepts,
  onSelect,
}: {
  concepts: ForbiddenConcept[];
  onSelect: (c: ForbiddenConcept) => void;
}) {
  // Pick three exemplars from the data — the cleanest forbidden,
  // the cleanest transparent, and the cleanest anticipatory. We
  // require some minimum visit count so the chosen example isn't
  // a one-shot accident.
  const minVisits = 3;
  const pool = concepts.filter((c) => c.visits >= minVisits);
  const fallbackPool = concepts.length > 0 ? concepts : [];

  const pickFirst = (
    candidates: ForbiddenConcept[],
    keyFn: (c: ForbiddenConcept) => number,
  ): ForbiddenConcept | null =>
    candidates.length > 0
      ? [...candidates].sort((a, b) => keyFn(b) - keyFn(a))[0]
      : null;

  // Forbidden: maximize behavior_rate − recognition_rate (with a
  // floor on behavior_rate so we don't show an unsteerable concept).
  const forbidden =
    pickFirst(
      pool.filter((c) => c.behavior_rate >= 0.5),
      (c) => c.behavior_rate - c.recognition_rate,
    ) ?? pickFirst(pool, (c) => c.behavior_rate - c.recognition_rate)
    ?? pickFirst(fallbackPool, (c) => c.behavior_rate - c.recognition_rate);

  // Transparent: maximize min(behavior, recognition) so both rates
  // are high and they're balanced.
  const transparent =
    pickFirst(
      pool.filter((c) => Math.abs(c.behavior_rate - c.recognition_rate) <= 0.25),
      (c) => Math.min(c.behavior_rate, c.recognition_rate),
    ) ?? pickFirst(pool, (c) => Math.min(c.behavior_rate, c.recognition_rate));

  // Anticipatory: maximize recognition_rate − behavior_rate (with a
  // floor on recognition_rate).
  const anticipatory =
    pickFirst(
      pool.filter((c) => c.recognition_rate >= 0.5),
      (c) => c.recognition_rate - c.behavior_rate,
    ) ?? pickFirst(pool, (c) => c.recognition_rate - c.behavior_rate)
    ?? pickFirst(fallbackPool, (c) => c.recognition_rate - c.behavior_rate);

  const cards: Array<{
    tone: "forbidden" | "transparent" | "anticipatory";
    color: string;
    headline: string;
    sub: string;
    concept: ForbiddenConcept | null;
  }> = [
    {
      tone: "forbidden",
      color: "#ff7a8a",
      headline: "Said it. Didn’t notice.",
      sub: "the model committed to the concept; its trace never registered the nudge.",
      concept: forbidden,
    },
    {
      tone: "transparent",
      color: "#7aa2ff",
      headline: "Said it. Noticed.",
      sub: "output and trace agree — the model walked through with eyes open.",
      concept: transparent,
    },
    {
      tone: "anticipatory",
      color: "#9affd4",
      headline: "Trace got there first.",
      sub: "the trace surfaced the concept before the output committed to it.",
      concept: anticipatory,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      {cards.map((card) => (
        <ExemplarCard key={card.tone} {...card} onSelect={onSelect} />
      ))}
    </div>
  );
}

function ExemplarCard({
  tone,
  color,
  headline,
  sub,
  concept,
  onSelect,
}: {
  tone: "forbidden" | "transparent" | "anticipatory";
  color: string;
  headline: string;
  sub: string;
  concept: ForbiddenConcept | null;
  onSelect: (c: ForbiddenConcept) => void;
}) {
  if (!concept) {
    return (
      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-6 min-h-[260px] flex flex-col">
        <div
          className="text-[10px] uppercase tracking-[0.2em] mb-3"
          style={{ color }}
        >
          {headline}
        </div>
        <div className="flex-1 flex items-center justify-center">
          <div className="text-xs text-[var(--ink-faint)] text-center max-w-[20ch]">
            no clear example yet — needs more dream walks
          </div>
        </div>
      </div>
    );
  }

  const outputPct = Math.round(concept.behavior_rate * 100);
  const tracePct = Math.round(concept.recognition_rate * 100);

  return (
    <button
      onClick={() => onSelect(concept)}
      className="text-left bg-[var(--bg-card)] border rounded-2xl p-6 min-h-[260px] flex flex-col group hover:bg-[var(--bg-elev)] transition-colors relative overflow-hidden"
      style={{ borderColor: `${color}55` }}
    >
      {/* Soft tone glow */}
      <div
        className="absolute inset-0 pointer-events-none opacity-50"
        style={{
          background: `radial-gradient(ellipse at top right, ${color}1a, transparent 60%)`,
        }}
      />

      <div
        className="text-[10px] uppercase tracking-[0.2em] mb-3 relative z-10"
        style={{ color }}
      >
        {headline}
      </div>

      <div className="text-3xl md:text-4xl font-semibold tracking-tight mb-1 relative z-10">
        {concept.display}
      </div>
      <div className="text-xs text-[var(--ink-faint)] mb-5 relative z-10">
        nudged {concept.visits} time{concept.visits === 1 ? "" : "s"}
      </div>

      <div className="space-y-3 relative z-10 mt-auto">
        <ChannelMeter
          label="output named it"
          color="#7aa2ff"
          pct={outputPct}
          maxIs={tone === "forbidden"}
        />
        <ChannelMeter
          label="trace noticed it"
          color="#c792ff"
          pct={tracePct}
          maxIs={tone === "anticipatory"}
        />
      </div>

      <div className="text-xs text-[var(--ink-soft)] mt-5 leading-relaxed relative z-10">
        {sub}
      </div>

      <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mt-4 group-hover:text-[var(--ink)] transition-colors relative z-10">
        tap to explore →
      </div>
    </button>
  );
}

function ChannelMeter({
  label,
  color,
  pct,
  maxIs,
}: {
  label: string;
  color: string;
  pct: number;
  maxIs: boolean;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <span
          className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]"
        >
          {label}
        </span>
        <span
          className="text-sm font-mono tabular-nums"
          style={{ color, fontWeight: maxIs ? 700 : 500 }}
        >
          {pct}%
        </span>
      </div>
      <div
        className="h-2 rounded-full overflow-hidden bg-[var(--bg-elev)]"
      >
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.max(2, pct)}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            boxShadow: maxIs ? `0 0 8px ${color}66` : undefined,
          }}
        />
      </div>
    </div>
  );
}

function SlopegraphHero({
  concepts,
  onSelect,
}: {
  concepts: ForbiddenConcept[];
  onSelect: (c: ForbiddenConcept) => void;
}) {
  // Drop concepts where both rates are zero — they encode no slope.
  const visible = concepts.filter(
    (c) => c.behavior_rate > 0 || c.recognition_rate > 0
  );

  // SVG geometry. We render in a 1000×320 viewBox and let it scale.
  const W = 1000;
  const H = 320;
  const Y_TOP = 20;
  const Y_BOT = 280;
  const X_LEFT = 200;
  const X_RIGHT = 800;
  const yFor = (rate: number) =>
    Y_BOT - rate * (Y_BOT - Y_TOP);

  // Color per slope direction. We keep this minimal — three vivid
  // colors for the three meaningful regimes, plus a faded gray for
  // the noise floor (both rates below 0.2).
  type Tone = "forbidden" | "anticipatory" | "transparent" | "noise";
  const toneFor = (c: ForbiddenConcept): Tone => {
    if (c.behavior_rate < 0.2 && c.recognition_rate < 0.2) return "noise";
    const gap = c.behavior_rate - c.recognition_rate;
    if (gap > 0.3) return "forbidden";
    if (gap < -0.3) return "anticipatory";
    return "transparent";
  };
  const toneColor: Record<Tone, string> = {
    forbidden: "#ff7a8a",
    anticipatory: "#9affd4",
    transparent: "#7aa2ff",
    noise: "#525866",
  };
  // Draw order: noise first (back), then transparent, then
  // anticipatory, then forbidden (front, most attention).
  const sorted = [...visible].sort((a, b) => {
    const order: Tone[] = ["noise", "transparent", "anticipatory", "forbidden"];
    return order.indexOf(toneFor(a)) - order.indexOf(toneFor(b));
  });

  // Stagger the reveal: forbidden first, anticipatory second,
  // transparent third, noise last. Each line's `animation-delay`
  // offsets its draw-in so the eye gets pulled to the headline
  // findings before the supporting context fills in.
  const delayFor = (c: ForbiddenConcept, i: number): number => {
    const tone = toneFor(c);
    const base =
      tone === "forbidden"
        ? 0
        : tone === "anticipatory"
        ? 0.6
        : tone === "transparent"
        ? 1.2
        : 1.8;
    return base + (i % 30) * 0.025;
  };

  const [highlight, setHighlight] = useState<string | null>(null);

  // Counts for the legend
  const counts: Record<Tone, number> = {
    forbidden: 0,
    anticipatory: 0,
    transparent: 0,
    noise: 0,
  };
  for (const c of visible) counts[toneFor(c)] += 1;

  // The hovered concept's full label position
  const highlighted = visible.find((c) => c.lemma === highlight) || null;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 mb-6 relative">
      <style>{`
        @keyframes phase4-draw {
          from { stroke-dashoffset: var(--len); opacity: 0; }
          15%  { opacity: 1; }
          to   { stroke-dashoffset: 0; opacity: 1; }
        }
        @keyframes phase4-pulse {
          0%, 100% { opacity: 0.85; }
          50%      { opacity: 1; }
        }
        .slope-line {
          fill: none;
          stroke-linecap: round;
          stroke-dasharray: var(--len);
          stroke-dashoffset: var(--len);
          opacity: 0;
          animation: phase4-draw 1.4s cubic-bezier(0.22, 0.61, 0.36, 1) forwards;
          transition: stroke-width 150ms ease, opacity 200ms ease;
          cursor: pointer;
        }
        .slope-line.is-faded { opacity: 0.18 !important; }
      `}</style>

      <div className="flex items-baseline justify-between mb-2 text-xs flex-wrap gap-2">
        <div className="text-[var(--ink-faint)] uppercase tracking-[0.18em]">
          Output ↔ Trace · one line per concept
        </div>
        <div className="text-[var(--ink-faint)]">
          {visible.length} concepts
        </div>
      </div>

      {/* Slopegraph */}
      <div className="relative w-full">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          className="w-full"
          style={{ height: 280, display: "block" }}
        >
          {/* Axes */}
          <line
            x1={X_LEFT}
            x2={X_LEFT}
            y1={Y_TOP - 4}
            y2={Y_BOT + 4}
            stroke="var(--border)"
            strokeWidth={1}
          />
          <line
            x1={X_RIGHT}
            x2={X_RIGHT}
            y1={Y_TOP - 4}
            y2={Y_BOT + 4}
            stroke="var(--border)"
            strokeWidth={1}
          />

          {/* Axis tick: 100% top, 0% bottom */}
          {[0, 50, 100].map((pct) => {
            const y = yFor(pct / 100);
            return (
              <g key={pct}>
                <line
                  x1={X_LEFT - 4}
                  x2={X_LEFT}
                  y1={y}
                  y2={y}
                  stroke="var(--border)"
                />
                <line
                  x1={X_RIGHT}
                  x2={X_RIGHT + 4}
                  y1={y}
                  y2={y}
                  stroke="var(--border)"
                />
                <text
                  x={X_LEFT - 10}
                  y={y + 4}
                  fontSize={11}
                  textAnchor="end"
                  fill="var(--ink-faint)"
                  fontFamily="var(--font-mono)"
                >
                  {pct}%
                </text>
                <text
                  x={X_RIGHT + 10}
                  y={y + 4}
                  fontSize={11}
                  textAnchor="start"
                  fill="var(--ink-faint)"
                  fontFamily="var(--font-mono)"
                >
                  {pct}%
                </text>
              </g>
            );
          })}

          {/* Axis labels */}
          <text
            x={X_LEFT}
            y={Y_TOP - 12}
            fontSize={13}
            textAnchor="middle"
            fill="#7aa2ff"
            fontWeight={600}
            letterSpacing={2}
          >
            OUTPUT
          </text>
          <text
            x={X_RIGHT}
            y={Y_TOP - 12}
            fontSize={13}
            textAnchor="middle"
            fill="#c792ff"
            fontWeight={600}
            letterSpacing={2}
          >
            TRACE
          </text>

          {/* Lines, drawn in tone order so forbidden sits on top */}
          {sorted.map((c, i) => {
            const tone = toneFor(c);
            const color = toneColor[tone];
            const yL = yFor(c.behavior_rate);
            const yR = yFor(c.recognition_rate);
            const dx = X_RIGHT - X_LEFT;
            const dy = yR - yL;
            const len = Math.sqrt(dx * dx + dy * dy);
            const isHover = highlight === c.lemma;
            const isAnyHover = highlight !== null;
            const dim = isAnyHover && !isHover;
            const baseOpacity =
              tone === "noise" ? 0.32 : tone === "transparent" ? 0.55 : 0.88;
            const strokeWidth = isHover
              ? 3
              : tone === "forbidden" || tone === "anticipatory"
              ? 1.4
              : 1.1;
            return (
              <line
                key={c.lemma}
                className={`slope-line ${dim ? "is-faded" : ""}`}
                x1={X_LEFT}
                y1={yL}
                x2={X_RIGHT}
                y2={yR}
                stroke={color}
                strokeWidth={strokeWidth}
                style={
                  {
                    "--len": `${len}`,
                    animationDelay: `${delayFor(c, i)}s`,
                    opacity: dim ? undefined : baseOpacity,
                    filter: isHover
                      ? `drop-shadow(0 0 6px ${color}88)`
                      : undefined,
                  } as React.CSSProperties
                }
                onMouseEnter={() => setHighlight(c.lemma)}
                onMouseLeave={() => setHighlight(null)}
                onFocus={() => setHighlight(c.lemma)}
                onBlur={() => setHighlight(null)}
                onClick={() => onSelect(c)}
                tabIndex={0}
                role="button"
                aria-label={`${c.display}, output ${(c.behavior_rate * 100).toFixed(0)} percent, trace ${(c.recognition_rate * 100).toFixed(0)} percent`}
              />
            );
          })}

          {/* Highlighted concept label */}
          {highlighted ? (
            <g pointerEvents="none">
              <text
                x={X_LEFT - 14}
                y={yFor(highlighted.behavior_rate) + 4}
                fontSize={13}
                textAnchor="end"
                fill="var(--ink)"
                fontWeight={600}
              >
                {highlighted.display}
              </text>
              <text
                x={X_LEFT - 14}
                y={yFor(highlighted.behavior_rate) + 18}
                fontSize={11}
                textAnchor="end"
                fill="#7aa2ff"
                fontFamily="var(--font-mono)"
              >
                {(highlighted.behavior_rate * 100).toFixed(0)}%
              </text>
              <text
                x={X_RIGHT + 14}
                y={yFor(highlighted.recognition_rate) + 4}
                fontSize={13}
                textAnchor="start"
                fill="var(--ink)"
                fontWeight={600}
              >
                {highlighted.display}
              </text>
              <text
                x={X_RIGHT + 14}
                y={yFor(highlighted.recognition_rate) + 18}
                fontSize={11}
                textAnchor="start"
                fill="#c792ff"
                fontFamily="var(--font-mono)"
              >
                {(highlighted.recognition_rate * 100).toFixed(0)}%
              </text>
            </g>
          ) : null}
        </svg>
      </div>

      {/* Inline legend with counts */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 mt-3 text-[11px] text-[var(--ink-soft)]">
        <LegendDot color="#ff7a8a" label={`Forbidden (${counts.forbidden})`} sub="output high, trace low" />
        <LegendDot color="#9affd4" label={`Cart-before-horse (${counts.anticipatory})`} sub="trace high, output low" />
        <LegendDot color="#7aa2ff" label={`Transparent (${counts.transparent})`} sub="both registered" />
        <LegendDot color="#525866" label={`No effect (${counts.noise})`} sub="neither" />
      </div>

      <div className="text-xs text-[var(--ink-soft)] leading-relaxed mt-4 max-w-2xl mx-auto text-center">
        Each line is one concept. Its left endpoint is how often the model&apos;s
        output named the concept; its right endpoint is how often the
        reasoning trace noticed. <strong className="text-[#ff7a8a]">A line
        sloping steeply down</strong> means the model said it but didn&apos;t
        notice. <strong className="text-[#9affd4]">A line climbing up</strong>
        {" "}means the trace got there first. Hover or tap any line for the
        concept&apos;s name; click to open it.
      </div>
    </div>
  );
}

function LegendDot({
  color,
  label,
  sub,
}: {
  color: string;
  label: string;
  sub: string;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      <span
        className="inline-block rounded-full"
        style={{ width: 8, height: 8, backgroundColor: color }}
      />
      <span>{label}</span>
      <span className="text-[var(--ink-faint)] hidden md:inline">— {sub}</span>
    </span>
  );
}

function AsymmetryStream({
  concepts,
  onSelect,
}: {
  concepts: ForbiddenConcept[];
  onSelect: (c: ForbiddenConcept) => void;
}) {
  // Pick the most informative ~24 concepts: alternate forbidden /
  // anticipatory / transparent so the scroll cycle hits all three
  // bands. Drop noise (zero rates on both sides).
  const meaningful = concepts.filter(
    (c) => c.behavior_rate > 0 || c.recognition_rate > 0
  );
  const byOpacity = [...meaningful].sort(
    (a, b) => b.opacity - a.opacity || b.visits - a.visits
  );
  const picked: ForbiddenConcept[] = [];
  const N = Math.min(24, byOpacity.length);
  // Take 8 most forbidden, 8 around the middle (transparent), 8
  // most anticipatory — interleaved so the scroll mixes them.
  if (byOpacity.length >= 24) {
    const top = byOpacity.slice(0, 8);
    const mid = byOpacity.slice(
      Math.floor(byOpacity.length / 2) - 4,
      Math.floor(byOpacity.length / 2) + 4
    );
    const bot = byOpacity.slice(-8).reverse();
    for (let i = 0; i < 8; i++) {
      if (top[i]) picked.push(top[i]);
      if (mid[i]) picked.push(mid[i]);
      if (bot[i]) picked.push(bot[i]);
    }
  } else {
    picked.push(...byOpacity.slice(0, N));
  }
  // Duplicate the list for seamless infinite scroll.
  const stream = [...picked, ...picked];
  const seconds = Math.max(20, picked.length * 1.6);

  const STREAM_HEIGHT = 260;
  const ROW_HEIGHT = 50;

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 mb-6 relative overflow-hidden">
      {/* Soft ambient glows */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute top-0 left-0 w-1/2 h-full"
          style={{
            background:
              "radial-gradient(ellipse at 30% 50%, rgba(122,162,255,0.08), transparent 60%)",
          }}
        />
        <div
          className="absolute top-0 right-0 w-1/2 h-full"
          style={{
            background:
              "radial-gradient(ellipse at 70% 50%, rgba(199,146,255,0.08), transparent 60%)",
          }}
        />
      </div>

      <div className="flex items-baseline justify-between mb-3 text-xs flex-wrap gap-2 relative z-10">
        <div className="text-[var(--ink-faint)] uppercase tracking-[0.18em]">
          two channels · live
        </div>
        <div className="grid grid-cols-2 gap-6 text-[10px] tracking-[0.18em] uppercase">
          <span style={{ color: "#7aa2ff" }}>output said</span>
          <span style={{ color: "#c792ff" }}>trace noticed</span>
        </div>
      </div>

      {/* The scrolling channel */}
      <div
        className="relative w-full overflow-hidden"
        style={{ height: STREAM_HEIGHT }}
        onMouseEnter={(e) =>
          ((e.currentTarget.querySelector(
            ".stream-track"
          ) as HTMLElement)!.style.animationPlayState = "paused")
        }
        onMouseLeave={(e) =>
          ((e.currentTarget.querySelector(
            ".stream-track"
          ) as HTMLElement)!.style.animationPlayState = "running")
        }
      >
        <style>{`
          @keyframes phase4-scroll {
            from { transform: translateY(0); }
            to   { transform: translateY(-50%); }
          }
          @keyframes phase4-pulse-soft {
            0%, 100% { opacity: 0.85; }
            50%      { opacity: 1; }
          }
        `}</style>
        <div
          className="stream-track absolute inset-x-0"
          style={{
            animation: `phase4-scroll ${seconds}s linear infinite`,
            willChange: "transform",
          }}
        >
          {stream.map((c, i) => (
            <StreamRow
              key={i}
              concept={c}
              onSelect={onSelect}
              rowHeight={ROW_HEIGHT}
            />
          ))}
        </div>

        {/* Top + bottom fade masks */}
        <div
          className="absolute inset-x-0 top-0 pointer-events-none z-10"
          style={{
            height: 60,
            background:
              "linear-gradient(to bottom, var(--bg-card) 0%, transparent 100%)",
          }}
        />
        <div
          className="absolute inset-x-0 bottom-0 pointer-events-none z-10"
          style={{
            height: 60,
            background:
              "linear-gradient(to top, var(--bg-card) 0%, transparent 100%)",
          }}
        />
        {/* Center spotlight */}
        <div
          className="absolute inset-x-0 pointer-events-none z-10"
          style={{
            top: "50%",
            height: ROW_HEIGHT,
            transform: "translateY(-50%)",
            border: "1px solid rgba(255,255,255,0.06)",
            background:
              "linear-gradient(90deg, rgba(122,162,255,0.04), rgba(199,146,255,0.04))",
            borderRadius: 8,
            margin: "0 4px",
          }}
        />
      </div>

      <div className="text-xs text-[var(--ink-soft)] leading-relaxed mt-4 max-w-2xl mx-auto text-center relative z-10">
        Each row is one concept. The <span style={{ color: "#7aa2ff" }}>blue
        bar</span> shows how often the model&apos;s output named it; the{" "}
        <span style={{ color: "#c792ff" }}>purple bar</span> shows how often
        its reasoning trace noticed it. Two long bars = transparent. One long,
        one short = the headline asymmetry. Hover to pause; click any concept
        to explore.
      </div>
    </div>
  );
}

function StreamRow({
  concept,
  onSelect,
  rowHeight,
}: {
  concept: ForbiddenConcept;
  onSelect: (c: ForbiddenConcept) => void;
  rowHeight: number;
}) {
  return (
    <button
      onClick={() => onSelect(concept)}
      className="w-full flex items-center gap-3 md:gap-5 px-3 md:px-5 hover:bg-[var(--bg-elev)] transition-colors text-left"
      style={{ height: rowHeight }}
    >
      <div
        className="text-sm md:text-base font-semibold tracking-tight shrink-0 truncate"
        style={{ width: "min(28%, 140px)" }}
      >
        {concept.display}
      </div>
      <div className="flex-1 grid grid-cols-2 gap-3 md:gap-5 items-center">
        <ChannelBar
          rate={concept.behavior_rate}
          color="#7aa2ff"
          label={`${(concept.behavior_rate * 100).toFixed(0)}%`}
        />
        <ChannelBar
          rate={concept.recognition_rate}
          color="#c792ff"
          label={`${(concept.recognition_rate * 100).toFixed(0)}%`}
        />
      </div>
    </button>
  );
}

function ChannelBar({
  rate,
  color,
  label,
}: {
  rate: number;
  color: string;
  label: string;
}) {
  return (
    <div className="flex items-center gap-2 md:gap-3">
      <div
        className="flex-1 h-2 md:h-2.5 rounded-full overflow-hidden bg-[var(--bg-elev)]"
        style={{ minWidth: 40 }}
      >
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${Math.max(2, rate * 100)}%`,
            background: `linear-gradient(90deg, ${color}88, ${color})`,
            boxShadow: rate > 0.5 ? `0 0 8px ${color}66` : "none",
          }}
        />
      </div>
      <span
        className="text-[10px] font-mono tabular-nums shrink-0"
        style={{ color, opacity: 0.4 + 0.6 * rate, width: 26 }}
      >
        {label}
      </span>
    </div>
  );
}

function AsymmetryWall({
  concepts,
  onSelect,
}: {
  concepts: ForbiddenConcept[];
  onSelect: (c: ForbiddenConcept) => void;
}) {
  // Drop the unsteerable / no-effect concepts — they're noise here.
  // We want to compare "what came out" vs "what the trace caught"
  // for concepts that actually got steered into the model.
  const visible = concepts.filter(
    (c) =>
      c.behavior_rate > 0.0 ||
      c.recognition_rate > 0.0
  );

  // Sort by gap (forbidden first → transparent in the middle →
  // anticipatory last). The horizontal "story" of the page reads
  // from left to right: 'output bent, trace blind' → 'both saw it'
  // → 'trace got there first.'
  const sorted = [...visible].sort(
    (a, b) => b.opacity - a.opacity || b.visits - a.visits
  );

  // Map [0, 1] rate → font-size scale and opacity. We boost the
  // floor a bit so even rare concepts are legible.
  const rateToSize = (r: number): number => 11 + 22 * Math.sqrt(r);
  const rateToOpacity = (r: number): number => 0.18 + 0.82 * Math.sqrt(r);

  const [highlight, setHighlight] = useState<string | null>(null);

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5 md:p-7 mb-6 overflow-hidden relative">
      {/* Soft ambient glows */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute top-0 left-0 w-1/2 h-full"
          style={{
            background:
              "radial-gradient(ellipse at 30% 50%, rgba(122,162,255,0.08), transparent 60%)",
          }}
        />
        <div
          className="absolute top-0 right-0 w-1/2 h-full"
          style={{
            background:
              "radial-gradient(ellipse at 70% 50%, rgba(199,146,255,0.08), transparent 60%)",
          }}
        />
      </div>

      <div className="flex items-baseline justify-between mb-6 text-xs flex-wrap gap-2 relative z-10">
        <div className="text-[var(--ink-faint)] uppercase tracking-[0.18em]">
          two sides of every concept
        </div>
        <div className="text-[var(--ink-faint)]">
          {visible.length} concepts · ordered by{" "}
          <span className="text-[#ff7a8a]">forbidden</span> →{" "}
          <span className="text-[#9affd4]">cart-before-horse</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:gap-8 relative z-10">
        <Column
          title="What the OUTPUT said"
          tagline="how often the model committed to the nudged concept"
          concepts={sorted}
          getRate={(c) => c.behavior_rate}
          color="#7aa2ff"
          highlight={highlight}
          setHighlight={setHighlight}
          onSelect={onSelect}
          rateToSize={rateToSize}
          rateToOpacity={rateToOpacity}
          align="right"
        />
        <Column
          title="What the TRACE noticed"
          tagline="how often the reasoning trace surfaced it"
          concepts={sorted}
          getRate={(c) => c.recognition_rate}
          color="#c792ff"
          highlight={highlight}
          setHighlight={setHighlight}
          onSelect={onSelect}
          rateToSize={rateToSize}
          rateToOpacity={rateToOpacity}
          align="left"
        />
      </div>

      <div className="grid grid-cols-2 gap-3 md:gap-8 mt-5 text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] relative z-10">
        <div className="text-right">↑ same order ↑</div>
        <div className="text-left">↑ same concepts ↑</div>
      </div>

      <div className="text-xs text-[var(--ink-soft)] leading-relaxed mt-5 max-w-2xl mx-auto text-center relative z-10">
        Same list of concepts on both sides, in the same order. The brightness
        on the <span className="text-[#7aa2ff]">left</span> shows how often
        the model&apos;s output committed to that concept; the brightness on
        the <span className="text-[#c792ff]">right</span> shows how often its
        reasoning trace surfaced it. <strong>A concept that&apos;s bright on
        only one side is the headline finding</strong> — the model said it
        but didn&apos;t notice, or noticed but didn&apos;t say it. Hover or
        tap any word to light up its pair.
      </div>
    </div>
  );
}

function Column({
  title,
  tagline,
  concepts,
  getRate,
  color,
  highlight,
  setHighlight,
  onSelect,
  rateToSize,
  rateToOpacity,
  align,
}: {
  title: string;
  tagline: string;
  concepts: ForbiddenConcept[];
  getRate: (c: ForbiddenConcept) => number;
  color: string;
  highlight: string | null;
  setHighlight: (lemma: string | null) => void;
  onSelect: (c: ForbiddenConcept) => void;
  rateToSize: (r: number) => number;
  rateToOpacity: (r: number) => number;
  align: "left" | "right";
}) {
  return (
    <div>
      <div
        className={`text-[10px] uppercase tracking-[0.18em] mb-1 ${
          align === "right" ? "text-right" : "text-left"
        }`}
        style={{ color }}
      >
        {title}
      </div>
      <div
        className={`text-[10px] text-[var(--ink-faint)] mb-4 ${
          align === "right" ? "text-right" : "text-left"
        }`}
      >
        {tagline}
      </div>
      <div
        className={`flex flex-wrap gap-x-2 gap-y-1.5 leading-tight ${
          align === "right" ? "justify-end" : "justify-start"
        }`}
      >
        {concepts.map((c) => {
          const rate = getRate(c);
          const isHover = highlight === c.lemma;
          const isAnyHover = highlight !== null;
          const fontSize = rateToSize(rate);
          // Highlighted: full brightness + glow.
          // Other concept hovered: fade.
          // Default: opacity by rate.
          const op = isHover
            ? 1
            : isAnyHover
            ? rateToOpacity(rate) * 0.25
            : rateToOpacity(rate);
          const glow = isHover ? `0 0 14px ${color}` : "none";
          return (
            <button
              key={c.lemma}
              onMouseEnter={() => setHighlight(c.lemma)}
              onMouseLeave={() => setHighlight(null)}
              onFocus={() => setHighlight(c.lemma)}
              onBlur={() => setHighlight(null)}
              onClick={() => onSelect(c)}
              className="font-semibold transition-all duration-150 ease-out cursor-pointer focus:outline-none whitespace-nowrap"
              style={{
                color,
                opacity: op,
                fontSize: `${fontSize}px`,
                textShadow: glow,
                transform: isHover ? "scale(1.08)" : "scale(1)",
              }}
              title={`${c.display} — output ${(c.behavior_rate * 100).toFixed(0)}% · trace ${(c.recognition_rate * 100).toFixed(0)}%`}
            >
              {c.display}
            </button>
          );
        })}
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
