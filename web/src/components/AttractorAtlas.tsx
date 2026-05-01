"use client";

import { useState } from "react";
import type { AttractorsFile, Attractor } from "@/lib/data";

const PAGE_SIZE = 4;

export function AttractorAtlas({ data }: { data: AttractorsFile }) {
  const total = data.attractors.length;
  const [page, setPage] = useState(0);
  const nPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const visible = data.attractors.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  return (
    <section className="px-6 py-16 max-w-6xl mx-auto">
      <header className="mb-6">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
          Phase 4 · gravity wells
        </div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
          Where dream walks fall
        </h2>
        <p className="text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          Most chains end at the same handful of concepts no matter where
          they started. We call those concepts <em>gravity wells</em>: a
          concept counts as a gravity well if two or more independent
          dream walks landed on it. For chains that ended by looping, the
          landing is the word the model emitted right before the loop
          closed. For chains that trailed off mid-thought, the landing is
          the last concept the chain was nudged toward.
        </p>
      </header>

      {total === 0 ? (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-12 text-center">
          <p className="text-[var(--ink-soft)]">
            No gravity wells yet. As more walks run, the same destination
            concepts will start appearing across multiple chains and land
            here.
          </p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {visible.map((a) => (
              <AttractorCard key={a.lemma} attractor={a} />
            ))}
          </div>
          {nPages > 1 && (
            <div className="flex items-center justify-between mt-6 text-xs text-[var(--ink-soft)]">
              <button
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--bg-card)] disabled:opacity-30 disabled:cursor-not-allowed hover:bg-[var(--bg-elev)] transition"
              >
                ← prev
              </button>
              <span className="font-mono tabular-nums text-[var(--ink-faint)]">
                {page + 1} / {nPages} &nbsp;·&nbsp; {total} wells
              </span>
              <button
                onClick={() => setPage((p) => Math.min(nPages - 1, p + 1))}
                disabled={page === nPages - 1}
                className="px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--bg-card)] disabled:opacity-30 disabled:cursor-not-allowed hover:bg-[var(--bg-elev)] transition"
              >
                next →
              </button>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function AttractorCard({ attractor: a }: { attractor: Attractor }) {
  const total = a.n_chains;
  const reasonRow = (label: string, count: number, color: string) =>
    count === 0 ? null : (
      <span
        className="text-[10px] uppercase tracking-[0.15em]"
        style={{ color }}
      >
        {count} {label}
      </span>
    );

  return (
    <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-5">
      <div className="flex items-baseline justify-between mb-3 flex-wrap gap-2">
        <div className="text-2xl font-semibold tracking-tight">
          {a.display}
        </div>
        <div className="text-xs font-mono tabular-nums text-[var(--ink-soft)]">
          {total} chain{total === 1 ? "" : "s"} landed here
        </div>
      </div>

      <div className="flex flex-wrap gap-3 mb-4">
        {reasonRow("looped to it", a.n_self_loop, "#7aa2ff")}
        {reasonRow("ended on it", a.n_coherence_break, "#c792ff")}
        {reasonRow("ran 20 to it", a.n_length_cap, "#9affd4")}
      </div>

      <div className="text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] mb-2">
        chains came from
      </div>
      <div className="flex flex-wrap gap-1.5">
        {a.sources.map((s) => (
          <span
            key={s.lemma}
            className="text-xs px-2 py-1 rounded-full bg-[var(--bg-elev)] text-[var(--ink-soft)] font-mono"
            title={
              s.n_chains > 1
                ? `${s.n_chains} chains seeded with ${s.display}`
                : `seeded with ${s.display}`
            }
          >
            {s.display}
            {s.n_chains > 1 ? (
              <span className="text-[var(--ink-faint)] ml-1">
                ×{s.n_chains}
              </span>
            ) : null}
          </span>
        ))}
      </div>
    </div>
  );
}
