"use client";

import type { Phase2Entry } from "@/lib/data";

/**
 * Decoder-cosine-neighbor graph for an SAE feature.
 *
 * Renders the parent feature in the center and its top-N decoder-cosine
 * neighbors in a single ring around it. Edge length encodes 1 - cosine
 * (closer = higher similarity), and tick mark on each spoke shows the
 * cosine value. Each node shows its short auto_interp label on hover.
 *
 * Pure SVG — no D3, no Recharts. The geometry is small (12 nodes max),
 * so a static layout is plenty.
 */
export function SaeNeighborGraph({ entry }: { entry: Phase2Entry }) {
  const sae = entry.sae;
  if (!sae || !sae.neighbors || sae.neighbors.length === 0) {
    return null;
  }
  const parent = {
    idx: sae.feature_idx,
    label: sae.auto_interp || "(unlabeled)",
  };
  // Use a square viewBox so the layout reads cleanly at any width.
  const W = 520;
  const H = 320;
  const cx = W / 2;
  const cy = H / 2;
  const RING_R = 130;

  // Sort neighbors by cosine descending so the strongest match is at the top.
  const sorted = [...sae.neighbors].sort((a, b) => b.cosine - a.cosine);
  const n = sorted.length;

  return (
    <div className="mt-5 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)] p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
        decoder-cosine neighborhood — top {n} features by W_dec similarity
      </div>
      <div className="flex justify-center">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          width="100%"
          height="auto"
          className="max-w-[520px]"
          role="img"
          aria-label={`Decoder-cosine neighbors of feature ${parent.idx}`}
        >
          {/* Spokes — drawn first so nodes render on top */}
          {sorted.map((nb, i) => {
            const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
            const x = cx + Math.cos(angle) * RING_R;
            const y = cy + Math.sin(angle) * RING_R;
            // Edge opacity tracks cosine — stronger neighbors look bolder.
            const opacity = 0.25 + 0.75 * Math.max(0, Math.min(1, nb.cosine));
            return (
              <line
                key={`spoke-${nb.feature_idx}`}
                x1={cx}
                y1={cy}
                x2={x}
                y2={y}
                stroke="var(--accent)"
                strokeOpacity={opacity}
                strokeWidth={1.2}
              />
            );
          })}

          {/* Neighbor nodes */}
          {sorted.map((nb, i) => {
            const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
            const x = cx + Math.cos(angle) * RING_R;
            const y = cy + Math.sin(angle) * RING_R;
            const labelX = cx + Math.cos(angle) * (RING_R + 30);
            const labelY = cy + Math.sin(angle) * (RING_R + 30);
            const anchor =
              labelX < cx - 8 ? "end" : labelX > cx + 8 ? "start" : "middle";
            return (
              <g key={`node-${nb.feature_idx}`}>
                <a
                  href={`https://www.neuronpedia.org/gemma-3-12b-it/31-gemmascope-2-res-262k/${nb.feature_idx}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <circle
                    cx={x}
                    cy={y}
                    r={6}
                    fill="var(--accent)"
                    fillOpacity={0.7}
                    stroke="var(--bg-card)"
                    strokeWidth={1.5}
                  >
                    <title>
                      #{nb.feature_idx} · cosine {nb.cosine.toFixed(3)}
                      {nb.auto_interp ? `\n${nb.auto_interp}` : ""}
                    </title>
                  </circle>
                </a>
                <text
                  x={labelX}
                  y={labelY}
                  textAnchor={anchor}
                  dominantBaseline="middle"
                  className="fill-[var(--ink-soft)]"
                  style={{ fontSize: 10 }}
                >
                  {nb.cosine.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Parent node — drawn last so it sits on top */}
          <circle
            cx={cx}
            cy={cy}
            r={11}
            fill="var(--success)"
            stroke="var(--bg-card)"
            strokeWidth={2}
          >
            <title>
              parent · #{parent.idx}
              {parent.label ? `\n${parent.label}` : ""}
            </title>
          </circle>
          <text
            x={cx}
            y={cy + 28}
            textAnchor="middle"
            className="fill-[var(--ink)]"
            style={{ fontSize: 11, fontWeight: 600 }}
          >
            #{parent.idx}
          </text>
        </svg>
      </div>
      <div className="text-xs text-[var(--ink-faint)] text-center mt-2 leading-relaxed">
        center node is the feature shown above; ring shows the {n} closest
        features in W_dec cosine space. Labels are cosine values; hover or
        click any node to see its auto-interp + open it on Neuronpedia.
      </div>
    </div>
  );
}
