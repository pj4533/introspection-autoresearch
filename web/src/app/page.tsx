import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { HowItWorks } from "@/components/HowItWorks";
import { Detections } from "@/components/Detections";
import { LayerCurve } from "@/components/LayerCurve";
import { Abliteration } from "@/components/Abliteration";
import { LivePhase2 } from "@/components/LivePhase2";
import { Footer } from "@/components/Footer";
import {
  loadSummary,
  loadDetections,
  loadLayerCurve,
  loadAbliterationComparison,
  loadPhase2Leaderboard,
  loadPhase2Activity,
} from "@/lib/data";

export default function Home() {
  const summary = loadSummary();
  const detections = loadDetections();
  const layerCurve = loadLayerCurve();
  const { variants } = loadAbliterationComparison();
  const leaderboard = loadPhase2Leaderboard();
  const activity = loadPhase2Activity();

  return (
    <main>
      <Nav />
      <Hero summary={summary} />
      <HowItWorks />
      <Detections detections={detections} />
      <LayerCurve data={layerCurve} />
      <Abliteration variants={variants} />
      <LivePhase2
        leaderboard={leaderboard}
        activity={activity}
        summary={summary}
      />
      <Footer />
    </main>
  );
}
