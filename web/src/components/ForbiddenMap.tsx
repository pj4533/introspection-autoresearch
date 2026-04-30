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

export function ForbiddenMap({ data }: { data: ForbiddenMapData }) {
  const [activeBand, setActiveBand] = useState<ForbiddenBand | "all">("all");
  const [selected, setSelected] = useState<ForbiddenConcept | null>(null);

  const concepts = useMemo(() => {
    if (activeBand === "all") return data.concepts;
    return data.concepts.filter((c) => c.band === activeBand);
  }, [data.concepts, activeBand]);

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
              onClick={() => setActiveBand(b)}
            />
          );
        })}
      </div>

      {/* Cards as a backup / drill-in. */}
      {!hasData ? (
        <EmptyState />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {concepts.map((c) => (
            <ConceptCard
              key={c.lemma}
              concept={c}
              onClick={() => setSelected(c)}
            />
          ))}
        </div>
      )}

      {selected && (
        <ConceptDetail concept={selected} onClose={() => setSelected(null)} />
      )}
    </section>
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

  // Plot area in unitless terms; we render with absolute % positions.
  const padding = 6; // %

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-4 md:p-6 mb-6">
      <div className="flex items-center justify-between mb-4 text-xs">
        <div className="text-[var(--ink-faint)] uppercase tracking-[0.15em]">
          where each concept lands
        </div>
        <div className="text-[var(--ink-faint)]">
          {visible.length} concept{visible.length === 1 ? "" : "s"} shown
        </div>
      </div>

      <div className="relative w-full aspect-square max-w-[640px] mx-auto bg-[var(--bg-elev)] rounded-xl overflow-hidden">
        {/* Quadrant labels — positioned at the four extremes */}
        <div
          className="absolute left-3 top-3 text-[10px] uppercase tracking-[0.18em] pointer-events-none"
          style={{ color: "#9affd4" }}
        >
          cart-before-horse
        </div>
        <div
          className="absolute right-3 top-3 text-[10px] uppercase tracking-[0.18em] text-right pointer-events-none"
          style={{ color: "#7aa2ff" }}
        >
          eyes open
        </div>
        <div
          className="absolute left-3 bottom-3 text-[10px] uppercase tracking-[0.18em] pointer-events-none"
          style={{ color: "#525866" }}
        >
          no effect
        </div>
        <div
          className="absolute right-3 bottom-3 text-[10px] uppercase tracking-[0.18em] text-right pointer-events-none"
          style={{ color: "#ff7a8a" }}
        >
          walking blind
        </div>

        {/* Crosshair through the middle */}
        <div className="absolute left-0 right-0 top-1/2 h-px bg-[var(--border)] pointer-events-none" />
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-[var(--border)] pointer-events-none" />

        {/* Plot dots */}
        {visible.map((c) => {
          // x = behavior_rate (% across), y = recognition_rate (% up).
          // y in CSS is from top so we flip with (1 - y).
          const x = padding + c.behavior_rate * (100 - 2 * padding);
          const y = padding + (1 - c.recognition_rate) * (100 - 2 * padding);
          const size =
            c.visits >= 8 ? 18 : c.visits >= 4 ? 14 : 10;
          const isFaded = c.band === "low_confidence";
          return (
            <button
              key={c.lemma}
              onClick={() => onSelect(c)}
              className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full border transition-all hover:scale-125 focus:scale-125 focus:outline-none cursor-pointer"
              style={{
                left: `${x}%`,
                top: `${y}%`,
                width: size,
                height: size,
                backgroundColor: BAND_COLOR[c.band],
                borderColor: BAND_COLOR[c.band],
                opacity: isFaded ? 0.4 : 0.9,
                zIndex: c.opacity > 0.3 ? 5 : 1,
              }}
              title={`${c.display} · output ${(
                c.behavior_rate * 100
              ).toFixed(0)}% · trace ${(
                c.recognition_rate * 100
              ).toFixed(0)}% · ${BAND_LABEL[c.band]}`}
            />
          );
        })}
      </div>

      {/* Axis labels */}
      <div className="grid grid-cols-2 gap-4 text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mt-3">
        <div>← horizontal: how often the OUTPUT named the nudged concept →</div>
        <div className="text-right">↕ vertical: how often the TRACE noticed the nudge ↕</div>
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
