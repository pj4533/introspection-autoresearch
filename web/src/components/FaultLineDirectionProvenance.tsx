"use client";

import type { Phase2Entry } from "@/lib/data";

/**
 * Provenance view for a Phase 2h fault-line steering direction.
 *
 * Each fault-line direction is built from ~50 positive prompts vs ~50
 * control prompts, encoded through the SAE, mean-differenced in feature
 * space, then projected back via W_dec. This component shows which
 * SAE features contributed most to the resulting direction, so a reader
 * can sanity-check that the direction is conceptually-shaped (e.g. for
 * the causality fault line, top features should describe causal
 * relations, not the word "because").
 *
 * Each top contributor is a Neuronpedia feature; clicking opens it.
 */
export function FaultLineDirectionProvenance({ entry }: { entry: Phase2Entry }) {
  const sae = entry.sae;
  if (!sae || !sae.top_features || sae.top_features.length === 0) {
    return null;
  }
  const top = sae.top_features.slice(0, 12);
  const maxWeight = Math.max(...top.map((f) => Math.abs(f.weight)));

  return (
    <div className="mt-5 rounded-lg bg-[var(--bg-elev)] border border-[var(--border)] p-4">
      <div className="text-xs uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
        direction provenance — top contributing SAE features
      </div>
      <div className="text-xs text-[var(--ink-soft)] mb-3 leading-relaxed">
        This direction was built from {sae.n_positive ?? "~"} positive
        prompts and {sae.n_control ?? "~"} control prompts, encoded
        through the SAE and mean-differenced in feature space. The
        features below contributed most strongly to the resulting
        residual-stream direction. Positive weights pulled toward the
        positive corpus; negative weights pulled away from it.
        {typeof sae.filtered_lexical_count === "number" &&
          sae.filtered_lexical_count > 0 && (
            <>
              {" "}
              <span className="text-[var(--ink-faint)]">
                ({sae.filtered_lexical_count} lexically-shaped features
                were zeroed out before projection.)
              </span>
            </>
          )}
      </div>
      <ul className="space-y-1.5">
        {top.map((f) => {
          const isPositive = f.weight >= 0;
          const barWidth = `${Math.round(
            (Math.abs(f.weight) / Math.max(maxWeight, 1e-9)) * 100
          )}%`;
          return (
            <li
              key={f.feature_idx}
              className="grid grid-cols-[auto_1fr_auto] items-center gap-2 text-xs"
            >
              <a
                href={`https://www.neuronpedia.org/gemma-3-12b-it/31-gemmascope-2-res-262k/${f.feature_idx}`}
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-[var(--accent)] hover:underline"
              >
                #{f.feature_idx}
              </a>
              <div className="relative h-4 flex items-center">
                <div
                  className={`absolute h-1.5 rounded-full ${
                    isPositive ? "bg-[var(--success)]" : "bg-[var(--warn)]"
                  }`}
                  style={{ width: barWidth, opacity: 0.55 }}
                />
                <span className="relative pl-2 text-[var(--ink-soft)] truncate">
                  {f.auto_interp || "(no auto-interp label)"}
                </span>
              </div>
              <span className="font-mono tabular-nums text-[var(--ink-faint)]">
                {isPositive ? "+" : "−"}
                {Math.abs(f.weight).toFixed(3)}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
