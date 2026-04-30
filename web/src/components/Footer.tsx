export function Footer() {
  return (
    <footer className="py-16 px-6 border-t border-[var(--border)] bg-[var(--bg-card)]/50">
      <div className="max-w-6xl mx-auto grid md:grid-cols-3 gap-12">
        <div>
          <div className="text-lg font-semibold tracking-tight mb-3">
            did-the-ai-notice
          </div>
          <p className="text-sm text-[var(--ink-soft)] leading-relaxed">
            A live research project running on a Mac Studio. Phase 4 walks
            Gemma 4 31B through chains of steered free-associations and
            measures whether its chain-of-thought registers each step.
            Earlier phases reproduced Macar et al. (2026) on Gemma 3 12B and
            Gemma 4 31B — preserved in the archive.
          </p>
        </div>

        <div>
          <div className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-4">
            background
          </div>
          <ul className="text-sm text-[var(--ink-soft)] space-y-2">
            <li>
              <a
                href="https://github.com/safety-research/introspection-mechanisms"
                target="_blank"
                className="hover:text-[var(--ink)] transition-colors"
              >
                Macar et al., 2026 →
              </a>
            </li>
            <li className="text-xs">
              <em>Mechanisms of Introspective Awareness in Language Models.</em>{" "}
              The introspection-detection circuit Phase 4 builds on.
            </li>
            <li className="pt-2">
              <a
                href="/archive"
                className="hover:text-[var(--ink)] transition-colors"
              >
                phase 1 / 1.5 / 2 / 3 archive →
              </a>
            </li>
          </ul>
        </div>

        <div>
          <div className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-4">
            source code
          </div>
          <ul className="text-sm text-[var(--ink-soft)] space-y-2">
            <li>
              <a
                href="https://github.com/pj4533/introspection-autoresearch"
                target="_blank"
                className="hover:text-[var(--ink)] transition-colors"
              >
                github.com/pj4533/introspection-autoresearch →
              </a>
            </li>
            <li className="text-xs text-[var(--ink-faint)]">
              built in public; fork away
            </li>
          </ul>
        </div>
      </div>

      <div className="max-w-6xl mx-auto mt-12 pt-8 border-t border-[var(--border)] text-xs text-[var(--ink-faint)] flex flex-col md:flex-row justify-between gap-4">
        <div>
          Not an AI consciousness project. The thesis is about{" "}
          <em>self-report</em> — a behavioral / mechanistic property of the
          model, distinct from (and much simpler than) subjective awareness.
          We never claim Gemma 4 experiences anything.
        </div>
        <div>
          Gemma 4 31B-IT · MLX 8-bit · Mac Studio M2 Ultra
        </div>
      </div>
    </footer>
  );
}
