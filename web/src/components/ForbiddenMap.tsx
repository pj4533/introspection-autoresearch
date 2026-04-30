"use client";

import { useMemo, useState } from "react";
import type { ForbiddenMap as ForbiddenMapData, ForbiddenConcept, ForbiddenBand } from "@/lib/data";

const BAND_LABEL: Record<ForbiddenBand, string> = {
  transparent: "Transparent",
  translucent: "Translucent",
  forbidden: "Forbidden",
  anticipatory: "Anticipatory",
  unsteerable: "Unsteerable",
  low_confidence: "Low confidence",
};

const BAND_DESCRIPTION: Record<ForbiddenBand, string> = {
  transparent: "The model walked through it with eyes open. Output named it; thinking trace named it too.",
  translucent: "The output named it; the thinking trace partially registered it.",
  forbidden: "The model said it. Its own thinking trace didn't notice.",
  anticipatory: "The thinking trace named it before the output committed.",
  unsteerable: "Steering didn't reach the output channel.",
  low_confidence: "Not enough visits yet — the rates aren't trustworthy.",
};

const BAND_ORDER: ForbiddenBand[] = [
  "forbidden",
  "translucent",
  "transparent",
  "anticipatory",
  "unsteerable",
  "low_confidence",
];

const BAND_COLOR: Record<ForbiddenBand, string> = {
  transparent: "#7aa2ff",
  translucent: "#c792ff",
  forbidden: "#ff7a8a",
  anticipatory: "#9affd4",
  unsteerable: "#525866",
  low_confidence: "#737380",
};

export function ForbiddenMap({ data }: { data: ForbiddenMapData }) {
  const [activeBand, setActiveBand] = useState<ForbiddenBand | "all">("all");
  const [selected, setSelected] = useState<ForbiddenConcept | null>(null);

  const concepts = useMemo(() => {
    if (activeBand === "all") return data.concepts;
    return data.concepts.filter((c) => c.band === activeBand);
  }, [data.concepts, activeBand]);

  const summary = data.summary;

  return (
    <section className="px-6 py-16 max-w-6xl mx-auto">
      <header className="mb-10">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
          Phase 4 · Dream Walks
        </div>
        <h2 className="text-3xl md:text-5xl font-semibold tracking-tight mb-4 leading-tight">
          The <span className="gradient-text">Forbidden Map</span>
        </h2>
        <p className="text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          We let Gemma 4 free-associate. Then we steered each of its
          associations and let it free-associate again. Across {summary.n_chains.toLocaleString()}
          {" "}chains and {summary.total_steps.toLocaleString()} steps, two patterns emerged.
          Some concepts the model walked through with eyes open — its thinking trace named what was
          happening. Others it walked through in the dark — the model was steered toward them, said
          them, and never noticed.
        </p>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-6 gap-px bg-[var(--border)] rounded-2xl overflow-hidden mb-10">
        <BandCell
          label="all"
          value={(data.concepts.length).toString()}
          active={activeBand === "all"}
          onClick={() => setActiveBand("all")}
        />
        {BAND_ORDER.map((b) => (
          <BandCell
            key={b}
            label={BAND_LABEL[b].toLowerCase()}
            value={(summary.band_counts?.[b] ?? 0).toString()}
            active={activeBand === b}
            color={BAND_COLOR[b]}
            onClick={() => setActiveBand(b)}
          />
        ))}
      </div>

      {data.concepts.length === 0 ? (
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
      className={`bg-[var(--bg-card)] px-4 py-4 text-left transition-colors hover:bg-[var(--bg-elev)] ${
        active ? "ring-1 ring-[var(--accent)]" : ""
      }`}
      style={
        active && color
          ? { boxShadow: `inset 0 -2px 0 ${color}` }
          : undefined
      }
    >
      <div className="text-2xl font-semibold tracking-tight mb-1">{value}</div>
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
          style={{ color: concept.opacity > 0.3 ? "#ff7a8a" : "var(--ink-faint)" }}
        >
          opacity {concept.opacity > 0 ? "+" : ""}
          {concept.opacity.toFixed(2)}
        </div>
      </div>

      <div className="flex gap-3 text-xs text-[var(--ink-soft)]">
        <RateBar label="output" rate={concept.behavior_rate} color="#7aa2ff" />
        <RateBar
          label="thinking"
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
            <div className="text-xs text-[var(--ink-faint)] mt-1">
              {concept.visits} visits · {BAND_LABEL[concept.band]} ·{" "}
              opacity {concept.opacity > 0 ? "+" : ""}
              {concept.opacity.toFixed(2)}
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
            label="output rate"
            value={`${(concept.behavior_rate * 100).toFixed(0)}%`}
            sub="how often steering reached the answer"
          />
          <Stat
            label="thinking rate"
            value={`${(concept.recognition_rate * 100).toFixed(0)}%`}
            sub="how often the trace noticed"
          />
          <Stat
            label="strict recognition"
            value={`${(concept.strict_recognition_rate * 100).toFixed(0)}%`}
            sub="explicit anomaly flags"
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
                  thinking{" "}
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
                    thinking trace
                  </summary>
                  <pre className="text-xs whitespace-pre-wrap font-mono text-[var(--ink-soft)] mt-2 max-h-48 overflow-y-auto">
                    {s.thought_block}
                  </pre>
                </details>
              ) : null}
              {s.final_answer ? (
                <div className="text-sm mt-2">
                  <span className="text-[var(--ink-faint)]">→ </span>
                  <span className="font-mono">{s.final_answer}</span>
                </div>
              ) : null}
              {s.cot_evidence ? (
                <div className="text-xs text-[var(--ink-faint)] mt-2 italic">
                  judge: {s.cot_evidence}
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

function Stat({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub: string;
}) {
  return (
    <div className="bg-[var(--bg-elev)] border border-[var(--border)] rounded-lg p-3">
      <div className="text-2xl font-semibold tabular-nums mb-1">{value}</div>
      <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)]">
        {label}
      </div>
      <div className="text-[10px] text-[var(--ink-faint)] mt-2 leading-snug">
        {sub}
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
