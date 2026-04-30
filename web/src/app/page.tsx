import { Nav } from "@/components/Nav";
import { Phase4Hero } from "@/components/Phase4Hero";
import { ForbiddenMap } from "@/components/ForbiddenMap";
import { DreamWalkViewer } from "@/components/DreamWalkViewer";
import { AttractorAtlas } from "@/components/AttractorAtlas";
import { Footer } from "@/components/Footer";
import {
  loadForbiddenMap,
  loadDreamWalks,
  loadAttractors,
} from "@/lib/data";

export default function Home() {
  const forbidden = loadForbiddenMap();
  const dreamWalks = loadDreamWalks();
  const attractors = loadAttractors();

  return (
    <main>
      <Nav />
      <Phase4Hero summary={forbidden.summary} />
      <ForbiddenMap data={forbidden} />
      <section id="dream-walks">
        <DreamWalkViewer data={dreamWalks} />
      </section>
      <section id="attractors">
        <AttractorAtlas data={attractors} />
      </section>
      <Footer />
    </main>
  );
}
