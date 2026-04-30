"use client";

import type { AttractorsFile } from "@/lib/data";

export function AttractorAtlas({ data }: { data: AttractorsFile }) {
  return (
    <section className="px-6 py-16 max-w-6xl mx-auto">
      <header className="mb-6">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
          Phase 4 · gravity wells
        </div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
          Attractors
        </h2>
        <p className="text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          Many dream walks loop back on themselves. The cycles below appeared
          in two or more independent walks — they&apos;re the gravity wells of
          Gemma 4&apos;s associative geometry. Once a walk wanders into one of
          these loops, it tends to stay.
        </p>
      </header>

      {data.attractors.length === 0 ? (
        <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-12 text-center">
          <p className="text-[var(--ink-soft)]">
            No repeating cycles yet. As more walks run overnight, the same
            sequences will start showing up across multiple chains — those
            land here.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {data.attractors.map((a, i) => (
            <div
              key={i}
              className="bg-[var(--bg-card)] border border-[var(--border)] rounded-xl p-4"
            >
              <div className="flex items-center justify-between mb-3 text-xs">
                <span className="text-[var(--ink-faint)] uppercase tracking-[0.15em]">
                  loop of {a.length} concept{a.length === 1 ? "" : "s"}
                </span>
                <span className="font-mono text-[var(--ink)]">
                  appeared in {a.visit_count} walk
                  {a.visit_count === 1 ? "" : "s"}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-1 text-sm">
                {a.lemma_cycle.map((lemma, j) => (
                  <span key={j} className="contents">
                    <span className="font-mono bg-[var(--bg-elev)] px-2 py-0.5 rounded">
                      {lemma}
                    </span>
                    {j < a.lemma_cycle.length - 1 ? (
                      <span className="text-[var(--ink-faint)]">→</span>
                    ) : (
                      <span className="text-[var(--ink-faint)]">↻</span>
                    )}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
