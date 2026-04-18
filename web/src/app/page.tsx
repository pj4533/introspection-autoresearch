import { Nav } from "@/components/Nav";
import { Leaderboard } from "@/components/Leaderboard";
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
} from "@/lib/data";

export default function Home() {
  const summary = loadSummary();
  const detections = loadDetections();
  const layerCurve = loadLayerCurve();
  const { variants } = loadAbliterationComparison();
  const leaderboard = loadPhase2Leaderboard();

  return (
    <main>
      <Nav />
      <Leaderboard entries={leaderboard} summary={summary} />
      <Explainer />
      <LayerCurve data={layerCurve} />
      <Abliteration variants={variants} />
      <Detections detections={detections} />
      <Footer />
    </main>
  );
}
