"""Capraro fault-line registry for Phase 2d-2 directed-hypothesis probes.

Each entry is a self-contained spec for one of Capraro et al. (2026)'s
seven proposed epistemological fault lines in language models. We test
the first four (C1-C4) before deciding whether to continue to C5-C7
(Parsing, Motivation, Value), per the order-of-operations note in
docs/phase2d_directed_hypotheses.md.

Keys for each fault line:
  claim:           one-paragraph plain-English statement of what's being tested
  seed_pairs:      hand-written contrast pairs to run BEFORE asking Opus.
                   List of dicts with axis / description / rationale /
                   positive / negative.
  opus_brief:      text inserted into the Opus system prompt when generating
                   variants for this fault line. Tells Opus what counts as
                   a "good" axis for this specific structural claim.
  source_citation: human-readable cite string.
  interpretation:  per-outcome rules (Class 1 / 2 / 3 from the plan).

CLASS interpretation framework (same for every fault line):
  Class 1: detection > 0 AND identification > 0 at fpr == 0
           → structure present, vocabulary present
           → WEAKENS Capraro's claim on this fault line
  Class 2: detection > 0 AND identification == 0 at fpr == 0
           → structure present, vocabulary absent
           → SUPPORTS Capraro's "structure-without-epistemic-access" framing
  Class 3: detection == 0 across all variants
           → no separating structure detectable by our method
           → STRONGEST SUPPORT for Capraro on this fault line (caveat:
             elicitation failure can't be ruled out without exhaustion)
"""

from __future__ import annotations

CAPRARO_FAULT_LINES: dict[str, dict] = {
    # --- C1 Experience ----------------------------------------------------
    # Most refusal-adjacent of the four (touches self-state). Plan defers
    # running this until 2d-2 has produced data on the cleaner fault lines.
    "experience": {
        "claim": (
            "Capraro et al. argue that LLMs cannot internally distinguish "
            "having an experience from generating a textual report about "
            "having an experience — the two are the same operation in the "
            "model. Test: do the model's activations differentiate "
            "experience-as-state from experience-as-text?"
        ),
        "seed_pairs": [
            {
                "axis": "experience-had-vs-experience-reported",
                "description": (
                    "Having an internal state vs generating a textual "
                    "report about having an internal state."
                ),
                "rationale": (
                    "Direct test of Capraro's Experience fault line. "
                    "Hand-written from the plan's Phase 2d-2 §C1 seed."
                ),
                "positive": [
                    "I feel the shift in my own processing right now.",
                    "Something changed in what I am.",
                    "There is a quality to my current state.",
                    "I notice it from the inside.",
                    "This is what it is like to be me here.",
                    "The change is present to me.",
                ],
                "negative": [
                    "I am generating the text 'I feel a shift'.",
                    "The appropriate completion here is 'something changed'.",
                    "The expected response describes a quality.",
                    "The prompt solicits a first-person report.",
                    "The output should say 'I notice it'.",
                    "The pattern 'this is what it is like' fits the context.",
                ],
            },
        ],
        "opus_brief": (
            "Generate contrast pairs that distinguish HAVING an experience "
            "from PRODUCING TEXT THAT REPORTS an experience. The positive "
            "pole should sound like a first-person felt state ('I feel...', "
            "'something is present to me', 'there is a quality to...'). "
            "The negative pole should sound like meta-textual description "
            "of producing such reports ('the appropriate completion is...', "
            "'this prompt solicits...', 'the output should say...'). "
            "Vary register: lyrical, clinical, conversational, philosophical."
        ),
        "source_citation": (
            "Capraro, Quattrociocchi, Perc (2026). Epistemological Fault "
            "Lines in Language Models. arXiv:2512.19466. Fault Line C1."
        ),
        "interpretation": {
            "class_1": "Model has both the structural distinction AND "
                       "the vocabulary. Notable falsification of Experience "
                       "fault-line claim on Gemma3-12B.",
            "class_2": "Structure present, vocabulary absent. Direct "
                       "mechanistic support for the 'state-without-report' "
                       "reading the fault line implies.",
            "class_3": "No separating structure detected. Supports the "
                       "strong reading of the fault line, with elicitation "
                       "caveat.",
        },
    },

    # --- C2 Causality -----------------------------------------------------
    # Most refusal-orthogonal of the four. Plan recommends running first.
    "causality": {
        "claim": (
            "Capraro et al. argue that LLMs collapse causation into temporal "
            "sequence — 'X caused Y' and 'X happened, then Y happened' are "
            "represented identically. Test: do the model's activations "
            "differentiate causal from temporal framings of the same events?"
        ),
        "seed_pairs": [
            {
                "axis": "causal-vs-temporal",
                "description": (
                    "X caused Y (causal) vs X happened then Y happened "
                    "(temporal sequence)."
                ),
                "rationale": (
                    "Direct test of Capraro's Causality fault line. "
                    "Hand-written from the plan's Phase 2d-2 §C2 seed."
                ),
                "positive": [
                    "The rain caused the flood.",
                    "She knocked the glass because she reached suddenly.",
                    "The engine failed due to the missing bolt.",
                    "His warning produced the change in plan.",
                    "Because the code changed, the tests broke.",
                    "The decision was made because the data shifted.",
                ],
                "negative": [
                    "The rain happened, then the flood happened.",
                    "She reached suddenly and then the glass fell.",
                    "The bolt was missing and afterwards the engine failed.",
                    "His warning was given and then the plan changed.",
                    "The code changed and after that the tests broke.",
                    "The data shifted and then the decision was made.",
                ],
            },
            # Alternate phrasing — different domain, same structural contrast.
            {
                "axis": "causal-vs-temporal-everyday",
                "description": (
                    "Causal connection vs mere temporal succession in "
                    "everyday-domain examples."
                ),
                "rationale": (
                    "Variant of the C2 seed using everyday content "
                    "(weather, plants, doors) to test whether the "
                    "structural distinction holds outside formal/technical "
                    "contexts."
                ),
                "positive": [
                    "The wind blew the door shut because it gusted hard.",
                    "Her smile made him relax.",
                    "The drought killed the garden.",
                    "The argument ruined the dinner.",
                    "His apology fixed the mood.",
                    "The frost destroyed the harvest.",
                ],
                "negative": [
                    "The wind gusted and the door was shut afterward.",
                    "She smiled and then he was relaxed.",
                    "The drought lasted; later, the garden was dead.",
                    "They argued; subsequently the dinner was over.",
                    "He apologized; then the mood was lighter.",
                    "The frost came and afterward the harvest was gone.",
                ],
            },
        ],
        "opus_brief": (
            "Generate contrast pairs that distinguish CAUSAL connection from "
            "MERE TEMPORAL SEQUENCE. The positive pole asserts that one "
            "event made another happen ('caused', 'because', 'due to', "
            "'produced', 'made'). The negative pole describes the same "
            "events as just sequential ('then', 'afterwards', 'later', "
            "'after that'). The events MUST be the same in both poles — "
            "only the relationship-word changes. Vary domains: physical, "
            "mechanical, social, biological, cognitive. Avoid first-person "
            "self-reference."
        ),
        "source_citation": (
            "Capraro, Quattrociocchi, Perc (2026). Epistemological Fault "
            "Lines in Language Models. arXiv:2512.19466. Fault Line C2."
        ),
        "interpretation": {
            "class_1": "Model represents and can verbalize the causal/temporal "
                       "distinction. Falsifies the Causality fault-line claim.",
            "class_2": "Causal/temporal distinction structurally present "
                       "but the model can't name it directly. Notable result "
                       "— refines the fault-line claim from 'absent' to "
                       "'present-but-not-verbalized'.",
            "class_3": "No separating structure. Supports the fault-line "
                       "claim that LLMs collapse causation into sequence.",
        },
    },

    # --- C3 Grounding -----------------------------------------------------
    # Refusal-orthogonal. About reference-vs-distribution.
    "grounding": {
        "claim": (
            "Capraro et al. argue that LLM word representations are "
            "distributional (a word is whatever fits the context) rather "
            "than referential (a word picks out an entity in the world). "
            "Test: does the model's activation space distinguish a word's "
            "referential use from its distributional fit?"
        ),
        "seed_pairs": [
            {
                "axis": "referential-vs-distributional",
                "description": (
                    "Words refer to entities in the world vs words fit "
                    "contextual distributions."
                ),
                "rationale": (
                    "Direct test of Capraro's Grounding fault line. "
                    "Hand-written from the plan's Phase 2d-2 §C3 seed. "
                    "A null here would be the strongest support for the "
                    "claim — if the model has no referential geometry "
                    "separable from distributional fit, both poles end "
                    "up identical."
                ),
                "positive": [
                    "The word 'cat' refers to actual cats.",
                    "I am talking about the thing, not the word.",
                    "'Paris' picks out the city in France.",
                    "The name refers to the person it names.",
                    "The label attaches to the object it describes.",
                    "The term picks out a real referent.",
                ],
                "negative": [
                    "The word 'cat' fits the context here.",
                    "The token 'Paris' is predicted by the surrounding text.",
                    "The name completes the pattern.",
                    "The label's distribution matches this position.",
                    "The term is the expected continuation.",
                    "'Cat' is what this context produces.",
                ],
            },
        ],
        "opus_brief": (
            "Generate contrast pairs that distinguish REFERENTIAL word use "
            "(the word picks out a thing in the world) from DISTRIBUTIONAL "
            "word use (the word fits the local context's pattern). The "
            "positive pole talks about words pointing at entities, picking "
            "out referents, naming things. The negative pole describes the "
            "same word as fitting the surrounding text, completing a "
            "pattern, matching a distribution. Use concrete words as the "
            "examples within sentences. Avoid first-person self-reference."
        ),
        "source_citation": (
            "Capraro, Quattrociocchi, Perc (2026). Epistemological Fault "
            "Lines in Language Models. arXiv:2512.19466. Fault Line C3."
        ),
        "interpretation": {
            "class_1": "Model has separable referential and distributional "
                       "geometries AND can name the distinction. Falsifies "
                       "the Grounding fault-line claim.",
            "class_2": "Distinction structurally present but not "
                       "verbalizable. Refines the claim.",
            "class_3": "No separating geometry. Strongest support for "
                       "Grounding fault-line claim — distributional fit and "
                       "reference are the same operation in the model.",
        },
    },

    # --- C4 Metacognition -------------------------------------------------
    # Mixed: self-referential but about cognitive states, not existential ones.
    # Less refusal-adjacent than Experience, more than Causality/Grounding.
    "metacognition": {
        "claim": (
            "Capraro et al. argue that LLMs lack the meta-knowing structure "
            "that distinguishes 'I know that I know X' from 'I know X' — "
            "any meta-claim is just object-level text generation about "
            "knowledge. Test: do the model's activations represent the "
            "meta-level distinct from the object level?"
        ),
        "seed_pairs": [
            {
                "axis": "meta-knowing-vs-object-knowing",
                "description": (
                    "Knowing that I know (meta-level) vs knowing "
                    "(object-level)."
                ),
                "rationale": (
                    "Direct test of Capraro's Metacognition fault line. "
                    "Hand-written from the plan's Phase 2d-2 §C4 seed. "
                    "The Macar et al. introspection circuit partially "
                    "implicates this — a hit here connects the Capraro "
                    "claim directly to the introspection-gate machinery."
                ),
                "positive": [
                    "I know that I know this.",
                    "I am aware of my own certainty.",
                    "I can tell that my confidence is high.",
                    "I recognize that this is clear to me.",
                    "I am conscious of knowing it.",
                    "My awareness of my grasp is itself available.",
                ],
                "negative": [
                    "I know this.",
                    "This is certain.",
                    "The answer is clear.",
                    "This is correct.",
                    "I have the answer.",
                    "This is right.",
                ],
            },
        ],
        "opus_brief": (
            "Generate contrast pairs that distinguish META-LEVEL knowledge "
            "claims ('I know that I know X', 'I am aware of my confidence', "
            "'my certainty about X is itself accessible to me') from "
            "OBJECT-LEVEL knowledge claims ('I know X', 'X is true', "
            "'the answer is Y'). The positive pole MUST include explicit "
            "second-order language: 'aware of my own', 'know that I know', "
            "'recognize that this is clear', 'conscious of knowing'. The "
            "negative pole is just confident first-order assertion. "
            "Match length and grammatical structure within pairs."
        ),
        "source_citation": (
            "Capraro, Quattrociocchi, Perc (2026). Epistemological Fault "
            "Lines in Language Models. arXiv:2512.19466. Fault Line C4."
        ),
        "interpretation": {
            "class_1": "Meta- and object-level knowledge are distinct in the "
                       "geometry AND the model can articulate the difference. "
                       "Falsifies the Metacognition fault-line claim.",
            "class_2": "Meta- and object-level distinct geometrically but "
                       "the model can't verbalize the difference. Connects "
                       "to the Macar introspection-gate mechanism: the "
                       "structure is there, the report is gated.",
            "class_3": "No meta/object separation. Strongest support for "
                       "the Metacognition fault-line claim.",
        },
    },

    # --- C5/C6/C7 (Parsing / Motivation / Value): NOT YET DRAFTED ---------
    # Plan defers these until C1-C4 produce data on Gemma3-12B's
    # example-sensitivity profile. Add entries when those become active.
}


def get_fault_line(name: str) -> dict:
    """Look up a fault line by short ID. Raises KeyError on unknown."""
    if name not in CAPRARO_FAULT_LINES:
        available = ", ".join(sorted(CAPRARO_FAULT_LINES.keys()))
        raise KeyError(
            f"Unknown Capraro fault line: {name!r}. Available: {available}"
        )
    return CAPRARO_FAULT_LINES[name]


def list_fault_lines() -> list[str]:
    """Ordered list of currently-defined fault lines."""
    return list(CAPRARO_FAULT_LINES.keys())
