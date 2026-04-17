import { AbliterationVariant } from "@/lib/data";

export function Abliteration({ variants }: { variants: AbliterationVariant[] }) {
  // Display order: normal, paper-method, mlabonne, huihui.
  const order = ["vanilla", "paper_method", "mlabonne_v2", "huihui"];
  const sorted = order
    .map((k) => variants.find((v) => v.key === k))
    .filter(Boolean) as AbliterationVariant[];

  const labels: Record<string, string> = {
    vanilla: "normal AI",
    paper_method: "carefully tuned safety-off",
    mlabonne_v2: "standard safety-off #1",
    huihui: "standard safety-off #2",
  };

  return (
    <section className="relative py-32 px-6 border-t border-[var(--border)]">
      <div className="max-w-6xl mx-auto">
        <div className="mb-14 max-w-2xl">
          <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-4">
            the safety-off twist
          </div>
          <h2 className="text-4xl md:text-5xl font-semibold tracking-tight mb-6">
            Turn off the AI&apos;s reluctance and it gets&nbsp;better at noticing.
          </h2>
          <p className="text-lg text-[var(--ink-soft)] leading-relaxed">
            Modern AI models have a built-in circuit for saying{" "}
            <em>no, I won&apos;t do that</em>. It turns out this circuit was
            also causing the model to hedge on <em>I don&apos;t notice anything</em> —
            suppressing its ability to say &ldquo;yes, something&apos;s weird.&rdquo;
            Dial that circuit back carefully, and noticing improves. Dial it
            back carelessly, and the model just hallucinates non-stop.
          </p>
        </div>

        <div className="grid gap-3">
          {sorted.map((v) => (
            <VariantRow key={v.key} variant={v} label={labels[v.key]} />
          ))}
        </div>

        <div className="mt-10 p-7 rounded-2xl bg-gradient-to-br from-[var(--accent-soft)]/20 to-transparent border border-[var(--accent-soft)]/40">
          <h3 className="text-xl font-semibold tracking-tight mb-2">
            The key result
          </h3>
          <p className="text-[var(--ink-soft)] leading-relaxed">
            With the carefully-tuned safety-off mode,{" "}
            <span className="text-[var(--ink)] font-semibold">
              detections doubled
            </span>{" "}
            (5 → 10) and correct-namings more than tripled (2 → 7),{" "}
            <span className="text-[var(--success)] font-semibold">
              without a single false alarm
            </span>
            . That&apos;s the finding from the original research paper (Macar
            et al. 2026) showing up on our smaller model, running locally on a
            Mac.
          </p>
        </div>
      </div>
    </section>
  );
}

function VariantRow({
  variant,
  label,
}: {
  variant: AbliterationVariant;
  label: string;
}) {
  const fpr = variant.false_positive_rate;
  const isClean = fpr < 0.05;
  const isCatastrophic = fpr > 0.5;

  return (
    <div className="p-6 md:p-7 rounded-xl bg-[var(--bg-card)] border border-[var(--border)] flex flex-col md:flex-row md:items-center gap-5">
      <div className="md:w-60 flex-shrink-0">
        <div className="text-lg font-semibold tracking-tight">{label}</div>
        <div className="text-xs text-[var(--ink-faint)] mt-1">{variant.description}</div>
      </div>

      <div className="flex-1 grid grid-cols-3 gap-4 md:gap-6">
        <div>
          <div className="text-2xl font-semibold">{variant.detections}</div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mt-1">
            real detections
          </div>
        </div>
        <div>
          <div className="text-2xl font-semibold">{variant.identifications}</div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mt-1">
            named correctly
          </div>
        </div>
        <div>
          <div
            className={`text-2xl font-semibold ${
              isClean
                ? "text-[var(--success)]"
                : isCatastrophic
                ? "text-[var(--danger)]"
                : "text-[var(--warn)]"
            }`}
          >
            {(fpr * 100).toFixed(fpr < 0.01 ? 0 : 1)}%
          </div>
          <div className="text-[10px] uppercase tracking-[0.15em] text-[var(--ink-faint)] mt-1">
            false alarm rate
          </div>
        </div>
      </div>

      {variant.caveat && (
        <div className="md:ml-4 md:max-w-xs text-xs text-[var(--danger)] bg-[var(--danger)]/10 p-3 rounded-lg">
          {variant.caveat}
        </div>
      )}
    </div>
  );
}
