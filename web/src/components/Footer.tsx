export function Footer() {
  return (
    <footer className="py-16 px-6 border-t border-[var(--border)] bg-[var(--bg-card)]/50">
      <div className="max-w-6xl mx-auto grid md:grid-cols-3 gap-12">
        <div>
          <div className="text-lg font-semibold tracking-tight mb-3">
            did-the-ai-notice
          </div>
          <p className="text-sm text-[var(--ink-soft)] leading-relaxed">
            A live research project running on a Mac Studio. Reproduces Macar
            et al. 2026 on the open-weights Gemma 3 12B model and extends the
            work with automated search for new steering directions.
          </p>
        </div>

        <div>
          <div className="text-xs uppercase tracking-[0.15em] text-[var(--ink-faint)] mb-4">
            the paper
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
            <li>
              <em>Mechanisms of Introspective Awareness in Language Models</em>
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
          Not an AI consciousness project. This measures a narrow capability:
          can the model detect engineered changes to its own internal state
          and accurately report on them? That&apos;s different from — and
          much simpler than — subjective awareness.
        </div>
        <div>
          Gemma 3 12B · local · Mac Studio M2 Ultra
        </div>
      </div>
    </footer>
  );
}
