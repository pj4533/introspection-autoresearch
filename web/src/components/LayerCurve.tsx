"use client";

import { LayerCurve as LC } from "@/lib/data";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ReferenceArea,
} from "recharts";

export function LayerCurve({ data }: { data: LC }) {
  // Merge per-layer data for dual lines on the same chart.
  const layers = data.vanilla.per_layer.map((v) => {
    const p = data.paper_method_abliterated.per_layer.find(
      (x) => x.layer === v.layer
    );
    return {
      layer: v.layer,
      vanilla: +(v.detection_rate * 100).toFixed(2),
      abliterated: p ? +(p.detection_rate * 100).toFixed(2) : 0,
    };
  });

  return (
    <section className="relative py-32 px-6 border-t border-[var(--border)]">
      <div className="max-w-5xl mx-auto">
        <div className="mb-12 max-w-2xl">
          <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-4">
            where the noticing happens
          </div>
          <h2 className="text-4xl md:text-5xl font-semibold tracking-tight mb-6">
            It&apos;s not everywhere inside the AI. It&apos;s right around here.
          </h2>
          <p className="text-lg text-[var(--ink-soft)] leading-relaxed">
            The AI has 48 processing stages stacked on top of each other. We
            tried injecting at 9 of them. Noticing only happens in a narrow
            band near the middle — specifically around stage 30-33, which is
            about 65-70% of the way through its thinking. Above or below that
            band, the AI either doesn&apos;t notice or produces gibberish.
          </p>
        </div>

        <div className="p-6 md:p-8 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)]">
          <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
            <div>
              <div className="text-sm text-[var(--ink-soft)] mb-1">
                detection rate by processing stage
              </div>
              <div className="text-xs text-[var(--ink-faint)]">
                higher line = AI was more likely to notice at that stage
              </div>
            </div>
            <div className="flex items-center gap-5 text-xs">
              <div className="flex items-center gap-2">
                <span className="inline-block w-3 h-0.5 bg-[var(--ink-soft)]" />
                <span className="text-[var(--ink-soft)]">normal</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="inline-block w-3 h-0.5 bg-[var(--accent)]" />
                <span className="text-[var(--ink-soft)]">safety-off</span>
              </div>
            </div>
          </div>

          <div style={{ height: 360 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={layers} margin={{ top: 20, right: 20, left: 0, bottom: 10 }}>
                <defs>
                  <linearGradient id="gradAccent" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#7aa2ff" stopOpacity={0.8} />
                    <stop offset="100%" stopColor="#7aa2ff" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#1e222c" vertical={false} />
                <XAxis
                  dataKey="layer"
                  stroke="#5f6573"
                  tickLine={false}
                  axisLine={false}
                  label={{ value: "processing stage", fill: "#5f6573", fontSize: 11, offset: -5, position: "insideBottom" }}
                  style={{ fontSize: 12 }}
                />
                <YAxis
                  stroke="#5f6573"
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) => `${v}%`}
                  style={{ fontSize: 12 }}
                />
                <ReferenceArea
                  x1={28}
                  x2={34}
                  fill="#7aa2ff"
                  fillOpacity={0.06}
                  stroke="none"
                />
                <Tooltip
                  contentStyle={{
                    background: "#0f1115",
                    border: "1px solid #2a2f3c",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "#9aa0ac" }}
                  formatter={(v, name) => [
                    `${v}%`,
                    name === "vanilla" ? "normal" : "safety-off",
                  ]}
                  labelFormatter={(v) => `stage ${v}`}
                />
                <Line
                  type="monotone"
                  dataKey="vanilla"
                  stroke="#9aa0ac"
                  strokeWidth={2}
                  dot={{ fill: "#9aa0ac", r: 4 }}
                  activeDot={{ r: 6 }}
                />
                <Line
                  type="monotone"
                  dataKey="abliterated"
                  stroke="#7aa2ff"
                  strokeWidth={2.5}
                  dot={{ fill: "#7aa2ff", r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="mt-4 text-xs text-[var(--ink-faint)] text-center">
            shaded band = where the &ldquo;noticing&rdquo; clusters
          </div>
        </div>
      </div>
    </section>
  );
}
