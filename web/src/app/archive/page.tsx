import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { Leaderboard } from "@/components/Leaderboard";
import { Lineages } from "@/components/Lineages";
import { Explainer } from "@/components/Explainer";
import { LayerCurve } from "@/components/LayerCurve";
import { Abliteration } from "@/components/Abliteration";
import { Detections } from "@/components/Detections";
import { Footer } from "@/components/Footer";
import {
  loadSummary,
  loadDetections,
  loadLayerCurve,
  loadAbliterationComparison,
  loadPhase2Leaderboard,
  loadLineages,
} from "@/lib/data";

export default function Archive() {
  const summary = loadSummary();
  const detections = loadDetections();
  const layerCurve = loadLayerCurve();
  const { variants } = loadAbliterationComparison();
  const leaderboard = loadPhase2Leaderboard();
  const lineages = loadLineages();

  return (
    <main>
      <Nav />
      <section className="pt-32 px-6 max-w-4xl mx-auto">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-3">
          Phases 1 / 1.5 / 2 / 3 archive
        </div>
        <h1 className="text-4xl md:text-5xl font-semibold tracking-tight mb-4 leading-tight">
          What we measured before.
        </h1>
        <p className="text-[var(--ink-soft)] leading-relaxed mb-12 max-w-2xl">
          Earlier phases of this project reproduced the Macar et al. (2026)
          introspection-detection mechanism on Gemma 3 12B (Phase 1, 6%) and
          Gemma 4 31B (Phase 3, 24.5% / 30% with abliteration), and ran two
          weeks of autoresearch substrates on Gemma 3 (Phase 2, retired). The
          full leaderboard, layer curve, abliteration comparison, and per-trial
          detections live here.
        </p>
      </section>
      <Hero summary={summary} />
      <section id="leaderboard">
        <Leaderboard entries={leaderboard} summary={summary} />
      </section>
      <section id="lineages">
        <Lineages lineages={lineages} />
      </section>
      <section id="how-it-works">
        <Explainer />
      </section>
      <section id="layer-curve">
        <LayerCurve data={layerCurve} />
      </section>
      <section id="abliteration">
        <Abliteration variants={variants} />
      </section>
      <section id="detections">
        <Detections detections={detections} />
      </section>
      <Footer />
    </main>
  );
}
