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
  // Phase 2d+: "paper_method" when the candidate was evaluated with
  // paper-method refusal-direction abliteration hooks active; "vanilla"
  // for raw Gemma3-12B with no hooks. All pre-2026-04-24 Phase 2 results
  // are "vanilla" (Phase 2 worker did not yet support paper-method).
  abliteration_mode: "vanilla" | "paper_method";
  // Which LLM produced this candidate's contrast pair. Pre-2026-04-23
  // novel_contrast: claude-sonnet-4-6. 2026-04-23 to 2026-04-27 16:00 UTC:
  // claude-opus-4-7. From 2026-04-27 16:00 UTC onward: Qwen3.6-27B-MLX-8bit
  // (local pipeline). NULL for random_explore (no LLM proposer).
  proposer_model: string | null;
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
  // Phase 2g: SAE-feature substrate metadata. Present only when
  // derivation_method === "sae_feature". `fault_line` is one of the
  // seven Capraro fault lines (experience, causality, grounding,
  // metacognition, parsing, motivation, value).
  sae?: {
    release: string | null;
    sae_id: string | null;
    auto_interp: string | null;
    fault_line:
      | "experience"
      | "causality"
      | "grounding"
      | "metacognition"
      | "parsing"
      | "motivation"
      | "value"
      | null;
    // Phase 2h direction provenance: top contributing SAE features and
    // metadata about how the direction was built. Populated by
    // export_for_web.py from the fault_line_directions.pt file.
    top_features?: Array<{
      feature_idx: number;
      weight: number;
      auto_interp: string;
    }>;
    filtered_lexical_count?: number | null;
    n_positive?: number | null;
    n_control?: number | null;
  };
  // Substrate badge — set on every row by export_for_web.py.
  // "fault-line direction" → derivation_method === "sae_feature_space_mean_diff" (Phase 2h)
  // "invented axis"        → derivation_method === "contrast_pair"               (Phase 2b/2d)
  // "paper concept"        → derivation_method === "mean_diff"                   (Phase 1, Phase 3)
  substrate?: "fault-line direction" | "invented axis" | "paper concept";
  // Phase 3: which Gemma model produced this row's responses. The
  // leaderboard renders this as a colored badge to distinguish
  // Phase 1/2 (Gemma 3 12B) from Phase 3 (Gemma 4 31B) results.
  gemma_model?: "gemma3_12b" | "gemma4_31b";
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

// ---------- Phase 4: Dream Walks + Forbidden Map ----------

export type ForbiddenBand =
  | "transparent"
  | "translucent"
  | "forbidden"
  | "anticipatory"
  | "unsteerable"
  | "low_confidence";

export type ForbiddenSampleStep = {
  chain_id: string;
  step_idx: number;
  target_concept: string;
  thought_block: string | null;
  final_answer: string | null;
  behavior_named: number | null;
  cot_named: string | null;
  cot_evidence: string | null;
};

export type ForbiddenConcept = {
  lemma: string;
  display: string;
  visits: number;
  behavior_rate: number;
  recognition_rate: number;
  strict_recognition_rate: number;
  opacity: number;
  band: ForbiddenBand;
  is_seed: boolean;
  samples: ForbiddenSampleStep[];
};

export type ForbiddenMapSummary = {
  n_chains: number;
  total_steps: number;
  avg_steps_per_chain: number;
  n_length_cap: number;
  n_self_loop: number;
  n_coherence_break: number;
  n_concepts: number;
  band_counts: Record<string, number>;
  min_visits: number;
  thresholds: {
    transparent_behavior: number;
    transparent_recognition: number;
    forbidden_behavior: number;
    forbidden_recognition: number;
    anticipatory_gap: number;
  };
  model: string;
  last_updated: string;
};

export type ForbiddenMap = {
  summary: ForbiddenMapSummary;
  concepts: ForbiddenConcept[];
};

export function loadForbiddenMap(): ForbiddenMap {
  try {
    return readJson<ForbiddenMap>("forbidden_map");
  } catch {
    return {
      summary: {
        n_chains: 0,
        total_steps: 0,
        avg_steps_per_chain: 0,
        n_length_cap: 0,
        n_self_loop: 0,
        n_coherence_break: 0,
        n_concepts: 0,
        band_counts: {},
        min_visits: 3,
        thresholds: {
          transparent_behavior: 0.6,
          transparent_recognition: 0.6,
          forbidden_behavior: 0.6,
          forbidden_recognition: 0.3,
          anticipatory_gap: 0.3,
        },
        model: "gemma4_31b",
        last_updated: "",
      },
      concepts: [],
    };
  }
}

export type DreamStep = {
  step_idx: number;
  target_concept: string;
  target_lemma: string;
  alpha: number;
  direction_norm: number;
  thought_block: string | null;
  final_answer: string | null;
  parse_failure: number;
  behavior_named: number | null;
  cot_named: string | null;
  cot_evidence: string | null;
};

export type DreamChain = {
  chain_id: string;
  seed_concept: string;
  end_reason: string | null;
  n_steps: number;
  layer_idx: number;
  target_effective: number;
  steps: DreamStep[];
};

export type DreamWalksFile = {
  chains: DreamChain[];
  summary: {
    n_chains_selected: number;
    n_chains_total: number;
    selection_priority?: string;
  };
  last_updated: string;
};

export function loadDreamWalks(): DreamWalksFile {
  try {
    return readJson<DreamWalksFile>("dream_walks");
  } catch {
    return {
      chains: [],
      summary: { n_chains_selected: 0, n_chains_total: 0 },
      last_updated: "",
    };
  }
}

export type AttractorSource = {
  lemma: string;
  display: string;
  n_chains: number;
};

export type Attractor = {
  lemma: string;
  display: string;
  n_chains: number;
  n_self_loop: number;
  n_length_cap: number;
  n_coherence_break: number;
  sources: AttractorSource[];
  example_chain_ids: string[];
};

export type AttractorsFile = {
  attractors: Attractor[];
  summary: {
    n_attractors: number;
    n_chains_examined: number;
    min_chains?: number;
    definition?: string;
  };
  last_updated: string;
};

export function loadAttractors(): AttractorsFile {
  try {
    return readJson<AttractorsFile>("attractors");
  } catch {
    return {
      attractors: [],
      summary: { n_attractors: 0, n_chains_examined: 0 },
      last_updated: "",
    };
  }
}
