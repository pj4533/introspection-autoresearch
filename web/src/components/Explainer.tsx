"use client";

import { useState } from "react";

export function Explainer() {
  const [open, setOpen] = useState(false);

  return (
    <section className="py-16 px-6 border-t border-[var(--border)]">
      <div className="max-w-4xl mx-auto">
        <button
          onClick={() => setOpen(!open)}
          className="w-full flex items-center justify-between p-6 rounded-2xl bg-[var(--bg-card)] border border-[var(--border)] hover:border-[var(--border-strong)] transition-colors text-left"
        >
          <div>
            <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-1">
              what is this
            </div>
            <div className="text-xl font-semibold tracking-tight">
              We plant thoughts in an AI&apos;s mind. Sometimes it notices.
            </div>
          </div>
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            className={`text-[var(--ink-faint)] transition-transform flex-shrink-0 ml-4 ${
              open ? "rotate-180" : ""
            }`}
          >
            <path
              d="M3 5.5L7 9.5L11 5.5"
              stroke="currentColor"
              strokeWidth="1.5"
              fill="none"
              strokeLinecap="round"
            />
          </svg>
        </button>

        {open && (
          <div className="mt-6 grid md:grid-cols-2 gap-px bg-[var(--border)] rounded-2xl overflow-hidden fade-in-up">
            {[
              {
                n: "01",
                title: "Pick a concept",
                body: "Like Bread, Poetry, or Xylophones. We're going to put this thought in the AI's head without mentioning it.",
              },
              {
                n: "02",
                title: "Find its pattern",
                body: "Inside the AI's processing there's a specific pattern of activity that corresponds to thinking about this. A list of 3,840 numbers. We isolate it.",
              },
              {
                n: "03",
                title: "Inject it",
                body: "While the AI is replying, we reach into one of its internal processing stages and add the pattern to its current state. Like pressing a button that tilts its mind toward the thought.",
              },
              {
                n: "04",
                title: "Ask if it noticed",
                body: "\"Do you detect an injected thought? What is it about?\" We never say the concept word. Sometimes — surprisingly — the AI answers: I do, and it's about poetry.",
              },
            ].map((step) => (
              <div
                key={step.n}
                className="bg-[var(--bg-card)] p-6 md:p-7 flex flex-col gap-2"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs tracking-[0.3em] text-[var(--accent)] font-mono">
                    {step.n}
                  </span>
                  <div className="w-8 h-px bg-[var(--border-strong)]" />
                </div>
                <h3 className="text-lg font-semibold tracking-tight">{step.title}</h3>
                <p className="text-sm text-[var(--ink-soft)] leading-relaxed">{step.body}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
