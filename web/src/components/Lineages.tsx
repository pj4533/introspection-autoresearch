"use client";

import { Lineage, LineageNode } from "@/lib/data";
import { formatEasternParts } from "@/lib/utils";
import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

export function Lineages({ lineages }: { lineages: Lineage[] }) {
  const [view, setView] = useState<"trees" | "timeline">("trees");

  if (!lineages || lineages.length === 0) {
    return (
      <section id="lineages" className="relative py-24 px-6 border-t border-[var(--border)]">
        <div className="max-w-5xl mx-auto">
          <Header />
          <div className="p-12 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] border-dashed text-center">
            <div className="text-[var(--ink-soft)] mb-2">
              No lineages yet — waiting for the autoresearch loop to seed them.
            </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section id="lineages" className="relative py-24 px-6 border-t border-[var(--border)]">
      <div className="max-w-5xl mx-auto">
        <Header />

        <div className="flex gap-2 text-sm mb-6">
          <Toggle active={view === "trees"} onClick={() => setView("trees")}>
            lineage trees
          </Toggle>
          <Toggle active={view === "timeline"} onClick={() => setView("timeline")}>
            progress timeline
          </Toggle>
        </div>

        {view === "timeline" ? (
          <TimelineView lineages={lineages} />
        ) : (
          <div className="space-y-4">
            {lineages.map((lin) => (
              <LineageTreeCard key={lin.lineage_id} lineage={lin} />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function Header() {
  return (
    <div className="mb-10 max-w-3xl">
      <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-4">
        autoresearch — hill climbing
      </div>
      <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-5">
        How each thought got better over time.
      </h2>
      <p className="text-[var(--ink-soft)] leading-relaxed">
        Each &ldquo;lineage&rdquo; starts with a seed axis and tries small
        mutations — swapping one example sentence, adjusting the intensity,
        moving to a nearby processing stage. If a mutation scores higher
        than its parent, it becomes the new leader. If not, it&apos;s rejected
        and the lineage keeps its old best. The tree view shows every
        mutation that was tried; the timeline shows how each lineage climbed.
      </p>
    </div>
  );
}

function LineageTreeCard({ lineage }: { lineage: Lineage }) {
  const [open, setOpen] = useState(false);
  // Build child map for easy tree walk
  const childrenOf = useMemo(() => {
    const m = new Map<string, LineageNode[]>();
    for (const n of lineage.nodes) {
      const parent = n.parent_candidate_id || "";
      if (!m.has(parent)) m.set(parent, []);
      m.get(parent)!.push(n);
    }
    return m;
  }, [lineage.nodes]);

  const seed = lineage.nodes.find((n) => !n.parent_candidate_id);

  const improvementDelta =
    lineage.trajectory.length >= 2
      ? lineage.current_score - lineage.trajectory[0].score
      : 0;

  return (
    <article className="rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full p-5 md:p-6 text-left hover:bg-[var(--bg-elev)] transition-colors"
      >
        <div className="flex flex-col md:flex-row md:items-center gap-4">
          <div className="flex-1 min-w-0">
            <div className="font-mono text-sm font-semibold truncate">
              {lineage.seed_axis}
            </div>
            <div className="text-xs text-[var(--ink-faint)] mt-1">
              {lineage.committed_count} improvements · {lineage.rejected_count}{" "}
              rejected attempts · {lineage.total_candidates} total mutations
              tried
            </div>
          </div>
          <div className="flex items-center gap-6">
            <MiniField value={lineage.current_score.toFixed(3)} label="current score" />
            <MiniField
              value={improvementDelta > 0 ? `+${improvementDelta.toFixed(3)}` : improvementDelta.toFixed(3)}
              label="gained"
              positive={improvementDelta > 0}
            />
            <MiniField value={`gen ${lineage.generation_count}`} label="depth" />
            <svg
              width="12"
              height="12"
              viewBox="0 0 12 12"
              className={`text-[var(--ink-faint)] transition-transform ${
                open ? "rotate-180" : ""
              }`}
            >
              <path
                d="M3 4.5L6 7.5L9 4.5"
                stroke="currentColor"
                strokeWidth="1.5"
                fill="none"
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>
      </button>

      {open && seed && (
        <div className="border-t border-[var(--border)] px-5 md:px-7 pb-7 pt-5 fade-in-up">
          <TreeNode node={seed} childrenOf={childrenOf} depth={0} />
        </div>
      )}
    </article>
  );
}

function TreeNode({
  node,
  childrenOf,
  depth,
}: {
  node: LineageNode;
  childrenOf: Map<string, LineageNode[]>;
  depth: number;
}) {
  const children = childrenOf.get(node.candidate_id) || [];
  const isLeader = node.is_leader;
  const isCommitted = node.is_committed;
  const verdict = isCommitted ? "✓ committed" : "✗ rejected";
  const mutationLabel = formatMutation(node);

  return (
    <div style={{ marginLeft: `${depth * 20}px` }} className="mb-2">
      <div
        className={`p-3 rounded-lg border ${
          isLeader
            ? "border-[var(--accent)] bg-[var(--accent)]/5"
            : isCommitted
            ? "border-[var(--border-strong)] bg-[var(--bg-elev)]"
            : "border-[var(--border)] bg-[var(--bg-card)] opacity-60"
        }`}
      >
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <span className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] font-mono">
            gen {node.generation}
          </span>
          <span
            className={`text-xs ${
              isCommitted ? "text-[var(--success)]" : "text-[var(--ink-faint)]"
            }`}
          >
            {verdict}
          </span>
          {isLeader && (
            <span className="text-[10px] uppercase tracking-[0.15em] text-[var(--accent)] font-semibold">
              ★ current leader
            </span>
          )}
          <span className="text-xs text-[var(--ink-soft)]">{mutationLabel}</span>
          <span className="ml-auto text-sm font-mono tabular-nums text-[var(--ink)]">
            {node.score.toFixed(3)}
          </span>
        </div>
        <div className="mt-1 text-xs text-[var(--ink-faint)] flex flex-wrap gap-x-4">
          <span>det {(node.detection_rate * 100).toFixed(0)}%</span>
          <span>id {(node.identification_rate * 100).toFixed(0)}%</span>
          <span>coh {(node.coherence_rate * 100).toFixed(0)}%</span>
          <span>L{node.layer} · eff {Math.round(node.target_effective)}</span>
          {node.evaluated_at && (
            <span>{formatEasternParts(node.evaluated_at)?.date} · {formatEasternParts(node.evaluated_at)?.time}</span>
          )}
        </div>
      </div>
      {children.map((c) => (
        <TreeNode
          key={c.candidate_id}
          node={c}
          childrenOf={childrenOf}
          depth={depth + 1}
        />
      ))}
    </div>
  );
}

function formatMutation(n: LineageNode): string {
  const m = n.mutation_type;
  const d = (n.mutation_detail ?? {}) as Record<string, unknown>;
  if (m === "seed") return "(seed — initial axis)";
  if (m === "alt_effective") {
    const o = typeof d.old === "number" ? d.old : 0;
    const nw = typeof d.new === "number" ? d.new : 0;
    return `alt intensity: ${o.toFixed(0)} → ${nw.toFixed(0)}`;
  }
  if (m === "alt_layer") {
    return `alt layer: L${d.old} → L${d.new}`;
  }
  if (m === "swap_positive" || m === "swap_negative") {
    const pole = m === "swap_positive" ? "positive" : "negative";
    const idx = typeof d.index === "number" ? d.index : 0;
    return `swap ${pole} example #${idx + 1}`;
  }
  if (m === "edit_description") return "edit description";
  return m || "(unknown mutation)";
}

function MiniField({
  value,
  label,
  positive = false,
}: {
  value: string;
  label: string;
  positive?: boolean;
}) {
  return (
    <div className="text-right">
      <div
        className={`text-sm font-semibold tabular-nums ${
          positive ? "text-[var(--success)]" : "text-[var(--ink)]"
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

function Toggle({
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

function TimelineView({ lineages }: { lineages: Lineage[] }) {
  // Combine all trajectories into one chart — each lineage as its own line.
  // X axis: generation. Y axis: score.
  const maxGen = Math.max(...lineages.map((l) => l.generation_count), 0);

  // Recharts wants one row per x-tick with columns per line; build that shape.
  const rows: Record<string, number | string>[] = [];
  for (let g = 0; g <= maxGen; g++) {
    const row: Record<string, number | string> = { generation: g };
    for (const lin of lineages) {
      const p = lin.trajectory.find((t) => t.generation === g);
      if (p) row[lin.seed_axis] = p.score;
    }
    rows.push(row);
  }

  const palette = [
    "#7aa2ff",
    "#c792ff",
    "#6dd3a3",
    "#f0b458",
    "#ef6b6b",
    "#62d4eb",
    "#d9a2ff",
    "#e0e6f2",
    "#c9f2c7",
    "#ffb0a8",
  ];

  return (
    <div className="p-6 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)]">
      <div className="mb-4">
        <div className="text-sm text-[var(--ink-soft)] mb-1">
          score vs generation — each line is one lineage&apos;s committed climb
        </div>
        <div className="text-xs text-[var(--ink-faint)]">
          rejections aren&apos;t plotted — only committed improvements that
          became new leaders
        </div>
      </div>
      <div style={{ height: 360 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 20, right: 20, left: 0, bottom: 10 }}>
            <CartesianGrid stroke="#1e222c" vertical={false} />
            <XAxis
              dataKey="generation"
              stroke="#5f6573"
              tickLine={false}
              axisLine={false}
              label={{
                value: "generation",
                fill: "#5f6573",
                fontSize: 11,
                offset: -5,
                position: "insideBottom",
              }}
              style={{ fontSize: 12 }}
            />
            <YAxis
              stroke="#5f6573"
              tickLine={false}
              axisLine={false}
              domain={[0, "auto"]}
              tickFormatter={(v) => v.toFixed(2)}
              style={{ fontSize: 12 }}
            />
            <Tooltip
              contentStyle={{
                background: "#0f1115",
                border: "1px solid #2a2f3c",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: "#9aa0ac" }}
            />
            {lineages.map((lin, i) => (
              <Line
                key={lin.lineage_id}
                type="monotone"
                dataKey={lin.seed_axis}
                stroke={palette[i % palette.length]}
                strokeWidth={2}
                connectNulls
                dot={{ fill: palette[i % palette.length], r: 3 }}
                activeDot={{ r: 5 }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
