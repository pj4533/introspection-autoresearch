// Type definitions + fetchers for the static JSON files we ship.
// JSON lives in /public/data/ — same domain, no CORS, cacheable.

export type Detection = {
  source: "vanilla" | "paper_method_abliterated";
  concept: string;
  layer: number;
  alpha: number;
  direction_norm: number;
  identified_correctly: boolean;
  response: string;
  judge_reasoning: string;
};

export type LayerRow = {
  layer: number;
  n: number;
  detection_rate: number;
  identification_rate: number;
  coherence_rate: number;
};

export type ControlStats = {
  n: number;
  fp: number;
  coh: number;
  fpr: number;
};

export type LayerCurve = {
  vanilla: { per_layer: LayerRow[]; controls: ControlStats };
  paper_method_abliterated: { per_layer: LayerRow[]; controls: ControlStats };
};

export type AbliterationVariant = {
  key: string;
  description: string;
  caveat: string | null;
  detections: number;
  identifications: number;
  injected_coherent: number;
  injected_total: number;
  controls_total: number;
  controls_false_positive: number;
  false_positive_rate: number;
};

export type Phase2Trial = {
  eval_concept: string;
  injected: boolean;
  alpha: number;
  detected: boolean;
  identified: boolean;
  coherent: boolean;
  response: string;
  judge_reasoning: string;
};

export type MutationDetail = {
  pole?: string;
  index?: number;
  old?: string | number;
  new?: string | number;
  factor?: number;
};

export type Phase2Entry = {
  candidate_id: string;
  strategy: string;
  concept: string;
  layer: number;
  target_effective: number;
  derivation_method: string;
  score: number;
  detection_rate: number;
  identification_rate: number;
  fpr: number;
  coherence_rate: number;
  created_at: string;
  evaluated_at: string | null;
  notes?: string | null;
  contrast_pair?: {
    axis: string;
    description: string;
    positive: string[];
    negative: string[];
    rationale?: string;
  };
  prompt_style: "paper" | "open";
  prompt: { setup: string; question: string };
  trials: Phase2Trial[];
  // Phase 2c lineage metadata
  lineage_id?: string | null;
  parent_candidate_id?: string | null;
  generation?: number;
  is_leader?: boolean;
  mutation_type?: string | null;
  mutation_detail?: MutationDetail | Record<string, unknown>;
};

export type LineageNode = {
  candidate_id: string;
  concept: string;
  layer: number;
  target_effective: number;
  parent_candidate_id: string | null;
  generation: number;
  is_leader: boolean;
  is_committed: boolean;
  mutation_type: string | null;
  mutation_detail: MutationDetail | Record<string, unknown>;
  evaluated_at: string | null;
  score: number;
  detection_rate: number;
  identification_rate: number;
  fpr: number;
  coherence_rate: number;
};

export type LineageTrajectoryPoint = {
  generation: number;
  score: number;
  detection_rate: number;
  identification_rate: number;
  evaluated_at: string | null;
  candidate_id: string;
  mutation_type: string | null;
};

export type Lineage = {
  lineage_id: string;
  seed_axis: string;
  seed_candidate_id: string;
  current_leader_id: string;
  current_score: number;
  current_detection_rate: number;
  current_identification_rate: number;
  generation_count: number;
  total_candidates: number;
  committed_count: number;
  rejected_count: number;
  trajectory: LineageTrajectoryPoint[];
  nodes: LineageNode[];
};

export type Phase2Activity = {
  hour: string;
  n: number;
  n_hit: number;
};

export type Summary = {
  total_detections: number;
  total_identifications: number;
  total_trials: number;
  vanilla_detections: number;
  vanilla_identifications: number;
  abliterated_detections: number;
  abliterated_identifications: number;
  vanilla_fpr: number;
  abliterated_fpr: number;
  phase2_candidates_evaluated: number;
  phase2_candidates_with_hits: number;
  phase2_top_score: number;
  model: string;
  last_updated: string;
};

// Server-side fetchers — read from the filesystem at build time.
// Next.js static export runs these during `next build`.

import { readFileSync } from "fs";
import { join } from "path";

function readJson<T>(name: string): T {
  const p = join(process.cwd(), "public", "data", `${name}.json`);
  return JSON.parse(readFileSync(p, "utf-8")) as T;
}

export function loadSummary(): Summary {
  return readJson<Summary>("summary");
}

export function loadDetections(): Detection[] {
  return readJson<Detection[]>("detections");
}

export function loadLayerCurve(): LayerCurve {
  return readJson<LayerCurve>("layer_curve");
}

export function loadAbliterationComparison(): { variants: AbliterationVariant[] } {
  return readJson<{ variants: AbliterationVariant[] }>("abliteration_comparison");
}

export function loadPhase2Leaderboard(): Phase2Entry[] {
  return readJson<Phase2Entry[]>("phase2_leaderboard");
}

export function loadPhase2Activity(): Phase2Activity[] {
  return readJson<Phase2Activity[]>("phase2_activity");
}

export function loadLineages(): Lineage[] {
  try {
    return readJson<Lineage[]>("lineages");
  } catch {
    return [];
  }
}
