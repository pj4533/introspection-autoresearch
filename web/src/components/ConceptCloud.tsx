"use client";

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import type { ForbiddenMap, ForbiddenConcept } from "@/lib/data";

const BAND_COLOR: Record<string, number> = {
  forbidden: 0xff5d6c,
  anticipatory: 0x5dc7ff,
  transparent: 0x6ee787,
  translucent: 0xc9b6ff,
  unsteerable: 0x6b6b6b,
  low_confidence: 0x4a4a4a,
};

type HoverInfo = {
  concept: ForbiddenConcept;
  x: number;
  y: number;
};

export function ConceptCloud({ data }: { data: ForbiddenMap }) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const concepts = data.concepts.filter((c) => c.visits >= 3);
    if (concepts.length === 0) return;

    const width = mount.clientWidth;
    const height = 520;

    const scene = new THREE.Scene();
    scene.background = null;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    let yaw = 0.6;
    let pitch = 0.3;
    let dist = 5.5;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    mount.appendChild(renderer.domElement);

    // axis frame: faint cube edges to ground the space
    const frameGeo = new THREE.BoxGeometry(2, 2, 2);
    const frameEdges = new THREE.EdgesGeometry(frameGeo);
    const frameMat = new THREE.LineBasicMaterial({
      color: 0x2a2a2a,
      transparent: true,
      opacity: 0.4,
    });
    const frame = new THREE.LineSegments(frameEdges, frameMat);
    scene.add(frame);

    // axis labels via tiny line markers (one bright stub per axis at the corner)
    const axisGeo = new THREE.BufferGeometry();
    const axisVerts = new Float32Array([
      -1, -1, -1, 1, -1, -1, // X (behavior)
      -1, -1, -1, -1, 1, -1, // Y (recognition)
      -1, -1, -1, -1, -1, 1, // Z (visits)
    ]);
    axisGeo.setAttribute("position", new THREE.BufferAttribute(axisVerts, 3));
    const axisMat = new THREE.LineBasicMaterial({
      color: 0x3a3a3a,
      transparent: true,
      opacity: 0.8,
    });
    scene.add(new THREE.LineSegments(axisGeo, axisMat));

    // build point cloud
    const positions: number[] = [];
    const colors: number[] = [];
    const sizes: number[] = [];

    const maxVisits = Math.max(...concepts.map((c) => c.visits));
    const logMax = Math.log(maxVisits + 1);

    const positionsArr: [number, number, number][] = [];

    for (const c of concepts) {
      const x = c.behavior_rate * 2 - 1; // [-1, 1]
      const y = c.recognition_rate * 2 - 1;
      const z = (Math.log(c.visits + 1) / logMax) * 2 - 1;
      positionsArr.push([x, y, z]);
      positions.push(x, y, z);

      const colorHex = BAND_COLOR[c.band] ?? 0x888888;
      const color = new THREE.Color(colorHex);
      colors.push(color.r, color.g, color.b);

      const baseSize = 1.6;
      const visitBoost = Math.log(c.visits + 1) / logMax;
      sizes.push(baseSize + visitBoost * 4.4);
    }

    const cloudGeo = new THREE.BufferGeometry();
    cloudGeo.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(positions), 3),
    );
    cloudGeo.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(colors), 3),
    );
    cloudGeo.setAttribute(
      "size",
      new THREE.BufferAttribute(new Float32Array(sizes), 1),
    );

    const vertexShader = `
      attribute float size;
      varying vec3 vColor;
      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (90.0 / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
      }
    `;
    const fragmentShader = `
      varying vec3 vColor;
      void main() {
        vec2 uv = gl_PointCoord - vec2(0.5);
        float d = length(uv);
        if (d > 0.5) discard;
        float core = smoothstep(0.5, 0.2, d);
        gl_FragColor = vec4(vColor, core);
      }
    `;

    const cloudMat = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
    });

    const cloud = new THREE.Points(cloudGeo, cloudMat);
    scene.add(cloud);

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points = { threshold: 0.06 };
    const pointerNDC = new THREE.Vector2();
    let pointerClient = { x: 0, y: 0 };
    let pointerActive = false;

    const onPointerMove = (e: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointerClient = { x: e.clientX, y: e.clientY };
      pointerNDC.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointerNDC.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      pointerActive = true;
      if (dragging) {
        const dx = e.clientX - dragLast.x;
        const dy = e.clientY - dragLast.y;
        yaw += dx * 0.005;
        pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.005));
        dragLast = { x: e.clientX, y: e.clientY };
      }
    };
    const onPointerLeave = () => {
      pointerActive = false;
      setHover(null);
    };

    let dragging = false;
    let dragLast = { x: 0, y: 0 };
    const onPointerDown = (e: PointerEvent) => {
      dragging = true;
      dragLast = { x: e.clientX, y: e.clientY };
    };
    const onPointerUp = () => {
      dragging = false;
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      dist = Math.max(2.5, Math.min(10, dist + e.deltaY * 0.003));
    };

    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerleave", onPointerLeave);
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("wheel", onWheel, { passive: false });

    let raf = 0;
    const animate = () => {
      if (!dragging) yaw += 0.0015;
      camera.position.x = dist * Math.cos(pitch) * Math.sin(yaw);
      camera.position.y = dist * Math.sin(pitch);
      camera.position.z = dist * Math.cos(pitch) * Math.cos(yaw);
      camera.lookAt(0, 0, 0);

      if (pointerActive && !dragging) {
        raycaster.setFromCamera(pointerNDC, camera);
        const hits = raycaster.intersectObject(cloud);
        if (hits.length > 0) {
          const idx = hits[0].index ?? 0;
          setHover({
            concept: concepts[idx],
            x: pointerClient.x,
            y: pointerClient.y,
          });
        } else {
          setHover(null);
        }
      }

      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    const onResize = () => {
      const w = mount.clientWidth;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("wheel", onWheel);
      cloudGeo.dispose();
      cloudMat.dispose();
      frameGeo.dispose();
      frameEdges.dispose();
      frameMat.dispose();
      axisGeo.dispose();
      axisMat.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
    };
  }, [data]);

  return (
    <section className="px-6 py-16 max-w-6xl mx-auto">
      <header className="mb-6">
        <div className="text-xs uppercase tracking-[0.2em] text-[var(--ink-faint)] mb-2">
          Phase 4 · concept geometry [experimental]
        </div>
        <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
          The shape of what the model can&apos;t notice
        </h2>
        <p className="text-[var(--ink-soft)] max-w-2xl leading-relaxed">
          Each dot is one concept. Position is its three-axis signature:
          how often the model spoke it (left → right), how often the trace
          flagged it (down → up), and how many chains visited it
          (back → front). Color is the band. Drag to orbit, scroll to
          zoom, hover to name a dot.
        </p>
      </header>

      <div className="bg-[var(--bg-card)] border border-[var(--border)] rounded-2xl p-4 relative">
        <div ref={mountRef} className="w-full" style={{ height: 520 }} />
        <div className="absolute top-4 left-4 text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] pointer-events-none space-y-0.5">
          <div>
            <span className="inline-block w-2 h-2 rounded-full bg-[#ff5d6c] mr-2 align-middle" />
            forbidden
          </div>
          <div>
            <span className="inline-block w-2 h-2 rounded-full bg-[#5dc7ff] mr-2 align-middle" />
            anticipatory
          </div>
          <div>
            <span className="inline-block w-2 h-2 rounded-full bg-[#6ee787] mr-2 align-middle" />
            transparent
          </div>
          <div>
            <span className="inline-block w-2 h-2 rounded-full bg-[#c9b6ff] mr-2 align-middle" />
            translucent
          </div>
          <div>
            <span className="inline-block w-2 h-2 rounded-full bg-[#6b6b6b] mr-2 align-middle" />
            unsteerable
          </div>
        </div>
        <div className="absolute bottom-4 right-4 text-[10px] uppercase tracking-[0.18em] text-[var(--ink-faint)] pointer-events-none">
          x: behavior · y: recognition · z: visits
        </div>
      </div>

      {hover && (
        <div
          className="fixed pointer-events-none z-50 bg-[var(--bg-card)] border border-[var(--border)] rounded-lg px-3 py-2 text-xs shadow-xl"
          style={{
            left: hover.x + 14,
            top: hover.y + 14,
          }}
        >
          <div className="font-semibold text-sm">{hover.concept.display}</div>
          <div className="text-[var(--ink-soft)] mt-1 space-y-0.5">
            <div>
              behavior {(hover.concept.behavior_rate * 100).toFixed(0)}%
            </div>
            <div>
              trace {(hover.concept.recognition_rate * 100).toFixed(0)}%
            </div>
            <div>visits {hover.concept.visits}</div>
            <div className="text-[var(--ink-faint)] uppercase tracking-wider text-[10px]">
              {hover.concept.band.replace("_", " ")}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
