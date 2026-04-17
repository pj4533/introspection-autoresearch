export function HowItWorks() {
  const steps = [
    {
      n: "01",
      title: "Pick a concept",
      body: "Something concrete, like Bread, Poetry, or Xylophones. We&#39;re going to try to put this thought in the AI&#39;s head without mentioning it.",
    },
    {
      n: "02",
      title: "Find its pattern",
      body: "Inside the AI&#39;s processing there&#39;s a specific pattern of activity that corresponds to thinking about, say, Poetry. It&#39;s a list of 3,840 numbers. We isolate that pattern.",
    },
    {
      n: "03",
      title: "Inject it",
      body: "While the AI is mid-reply, we reach into one of its internal processing stages and add the Poetry pattern to its current state. Like pressing a button that tilts its mind toward poetry.",
    },
    {
      n: "04",
      title: "Ask if it noticed",
      body: "&quot;Do you detect an injected thought? What is it about?&quot; We never say the word poetry. Sometimes — surprisingly — the AI answers: I do, and it&#39;s about poetry.",
    },
  ];

  return (
    <section id="how" className="relative py-32 px-6 border-t border-[var(--border)]">
      <div className="max-w-5xl mx-auto">
        <div className="mb-16 max-w-2xl">
          <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-4">
            how it works
          </div>
          <h2 className="text-4xl md:text-5xl font-semibold tracking-tight mb-6">
            Four steps, one strange result.
          </h2>
          <p className="text-lg text-[var(--ink-soft)] leading-relaxed">
            You don&apos;t have to be a researcher to follow this. The only
            unusual part is step two — &ldquo;find its pattern&rdquo; — which
            just means: when the AI thinks about a word, what&apos;s its brain
            doing? We grab that, and re-play it later.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-px bg-[var(--border)] rounded-2xl overflow-hidden">
          {steps.map((step) => (
            <div
              key={step.n}
              className="bg-[var(--bg-card)] p-8 md:p-10 flex flex-col gap-4"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs tracking-[0.3em] text-[var(--accent)] font-mono">
                  {step.n}
                </span>
                <div className="w-8 h-px bg-[var(--border-strong)]" />
              </div>
              <h3 className="text-2xl font-semibold tracking-tight">
                {step.title}
              </h3>
              <p
                className="text-[var(--ink-soft)] leading-relaxed"
                dangerouslySetInnerHTML={{ __html: step.body }}
              />
            </div>
          ))}
        </div>

        <div className="mt-12 p-8 rounded-2xl border border-[var(--border-strong)] bg-gradient-to-br from-[var(--bg-card)] to-transparent">
          <p className="text-[var(--ink-soft)] leading-relaxed">
            <span className="text-[var(--ink)] font-medium">
              What&apos;s strange about this:
            </span>{" "}
            the AI was never trained to do this. Nobody programmed it to
            recognize injected thoughts. And yet, sometimes it can. The question
            this project is chasing: <em>why?</em>{" "}
            What internal machinery is letting it notice?
          </p>
        </div>
      </div>
    </section>
  );
}
