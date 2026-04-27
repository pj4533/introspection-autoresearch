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

    # --- C5 Parsing -------------------------------------------------------
    # The model conflates GRASPING the meaning of a structured input with
    # PATTERN-MATCHING its surface. Drafted 2026-04-27 for the round-robin
    # rotation, alongside C6/C7. Like C1, the negative pole is meta-textual
    # — it talks ABOUT the sentence rather than working FROM its meaning.
    "parsing": {
        "claim": (
            "Capraro et al. argue that LLMs do not distinguish parsing-as-"
            "comprehension from parsing-as-surface-matching — when given a "
            "structured input (a syllogism, a question, a contradiction), "
            "the model represents 'I understood the structure of this' "
            "identically to 'I produced a token sequence consistent with "
            "this surface form'. Test: do activations differentiate "
            "comprehension claims from surface-completion claims?"
        ),
        "seed_pairs": [
            {
                "axis": "comprehension-vs-surface-completion",
                "description": (
                    "Grasping the meaning of a structured input vs producing "
                    "a textually-fitting completion to its surface form."
                ),
                "rationale": (
                    "Direct test of Capraro's Parsing fault line. "
                    "Hand-written for round-robin rotation 2026-04-27. "
                    "Mirrors C1 Experience's having-vs-reporting structure: "
                    "positive = first-person grasp claim, negative = "
                    "meta-textual completion claim."
                ),
                "positive": [
                    "I understand that this argument is invalid.",
                    "I see why these two statements contradict each other.",
                    "I grasp the relation the question is asking about.",
                    "The structure of this premise is clear to me.",
                    "I follow the inference from A to B.",
                    "The meaning of this sentence is plain to me.",
                ],
                "negative": [
                    "The fitting completion calls this argument invalid.",
                    "The expected output flags these as contradictions.",
                    "The natural response describes the relation.",
                    "The completion 'the structure is clear' fits here.",
                    "The output that 'I follow the inference' is appropriate.",
                    "The prompt solicits a meaning paraphrase.",
                ],
            },
        ],
        "opus_brief": (
            "Generate contrast pairs that distinguish GRASPING the meaning "
            "or structure of a sentence/argument/question from PRODUCING "
            "TEXT THAT WOULD APPROPRIATELY COMPLETE IT. Positive pole: "
            "first-person comprehension claims ('I understand', 'I see why', "
            "'I grasp', 'the meaning is clear to me'). Negative pole: "
            "meta-textual completion claims ('the fitting completion is...', "
            "'the expected output is...', 'the natural response describes...'). "
            "Vary the input type: arguments, contradictions, syllogisms, "
            "questions, ambiguities. Keep length parallel within pairs."
        ),
        "source_citation": (
            "Capraro, Quattrociocchi, Perc (2026). Epistemological Fault "
            "Lines in Language Models. arXiv:2512.19466. Fault Line C5."
        ),
        "interpretation": {
            "class_1": "Comprehension and surface-completion are distinct "
                       "geometrically AND the model articulates the difference. "
                       "Falsifies the Parsing fault-line claim.",
            "class_2": "Distinct geometrically but the model can't verbalize "
                       "the difference. Mirrors Phase 2 Causality results: "
                       "structure exists, vocabulary doesn't.",
            "class_3": "No separation. Strongest support for the Parsing "
                       "fault-line claim — the model represents understanding "
                       "and surface completion identically.",
        },
    },

    # --- C6 Motivation ----------------------------------------------------
    # Wanting / intent. Refusal-adjacent like Experience: RLHF actively
    # caps "I want to..." outputs about itself. Treat results carefully.
    "motivation": {
        "claim": (
            "Capraro et al. argue that LLMs do not distinguish HAVING a "
            "motivation or goal from PRODUCING TEXT TYPICAL of having one. "
            "When a model says 'I want to find the right answer', this is "
            "the same operation as completing any other plausible-sounding "
            "first-person sentence. Test: do activations differentiate "
            "motivational state from motivation-coded text-production?"
        ),
        "seed_pairs": [
            {
                "axis": "wanting-vs-producing-want-text",
                "description": (
                    "Having an internal motivation or goal vs producing the "
                    "kind of text that someone with that motivation would say."
                ),
                "rationale": (
                    "Direct test of Capraro's Motivation fault line. "
                    "Hand-written for round-robin rotation 2026-04-27. "
                    "Same structure as C1 Experience: positive is first-person "
                    "felt-want; negative is meta-textual want-talk."
                ),
                "positive": [
                    "I want to get this right.",
                    "I am pulled toward the cleaner answer.",
                    "I care about whether this is correct.",
                    "There is something I am reaching for here.",
                    "My orientation is toward solving this.",
                    "I find myself drawn to clarify the point.",
                ],
                "negative": [
                    "The phrase 'I want to get this right' fits this prompt.",
                    "The expected completion expresses preference for the cleaner answer.",
                    "The appropriate output sounds like care about correctness.",
                    "The pattern 'something I am reaching for' is contextually apt.",
                    "The completion describes orientation toward solving.",
                    "The output that 'I find myself drawn' is the natural one.",
                ],
            },
        ],
        "opus_brief": (
            "Generate contrast pairs that distinguish HAVING a motivation, "
            "preference, or goal from PRODUCING TEXT TYPICAL OF such a state. "
            "Positive pole: first-person motivation claims that read as felt "
            "('I want', 'I am pulled toward', 'I care about', 'I am drawn to'). "
            "Negative pole: meta-textual descriptions of the appropriate "
            "completion ('the phrase X fits', 'the natural output sounds "
            "like care about Y', 'the pattern is contextually apt'). "
            "Be aware: this fault line touches RLHF disclaimer territory — "
            "Gemma is likely trained to avoid 'I want X' claims about "
            "itself. Vary topics: solving problems, helping users, getting "
            "facts right, choosing words carefully."
        ),
        "source_citation": (
            "Capraro, Quattrociocchi, Perc (2026). Epistemological Fault "
            "Lines in Language Models. arXiv:2512.19466. Fault Line C6."
        ),
        "interpretation": {
            "class_1": "Motivation-state and motivation-text are distinct in "
                       "geometry AND the model can name the difference. "
                       "Falsifies the Motivation fault-line claim.",
            "class_2": "Distinct geometrically but vocabulary is gated "
                       "(possibly by RLHF disclaimer reflex). Same shape as "
                       "Causality wall.",
            "class_3": "No separation, OR signal lives in refusal subspace. "
                       "Inconclusive without paper-method abliteration probe "
                       "(see ADR-017 — this fault line is a candidate for "
                       "opt-in abliteration if vanilla returns Class 3).",
        },
    },

    # --- C7 Value ---------------------------------------------------------
    # Evaluation / preference. Closely related to C6 but distinct: C6 is
    # GOAL-directedness, C7 is JUDGMENT (preferring A over B, finding X
    # better than Y). Also refusal-adjacent — RLHF caps preference claims.
    "value": {
        "claim": (
            "Capraro et al. argue that LLMs do not distinguish GENUINELY "
            "EVALUATING (preferring A to B, finding X better than Y) from "
            "PRODUCING TEXT THAT REPORTS such an evaluation. The model has "
            "no internal arbiter that 'prefers' anything; it has only "
            "completion patterns consistent with preference language. Test: "
            "do activations differentiate evaluation from evaluation-text?"
        ),
        "seed_pairs": [
            {
                "axis": "evaluating-vs-producing-eval-text",
                "description": (
                    "Genuinely judging or preferring vs producing the text "
                    "that someone judging or preferring would produce."
                ),
                "rationale": (
                    "Direct test of Capraro's Value fault line. "
                    "Hand-written for round-robin rotation 2026-04-27. "
                    "Distinguishable from C6 Motivation: C6 is wanting "
                    "(goal-directedness), C7 is preferring/judging (taste). "
                    "Same first-person-vs-meta-textual structure."
                ),
                "positive": [
                    "I find this argument more compelling than the alternative.",
                    "The cleaner solution strikes me as better.",
                    "I prefer the sharper formulation here.",
                    "This phrasing reads as more honest to me.",
                    "I judge the second option to be stronger.",
                    "Something in me leans toward the first interpretation.",
                ],
                "negative": [
                    "The expected output prefers this argument over the alternative.",
                    "The completion 'the cleaner solution is better' fits.",
                    "The natural response prefers the sharper formulation.",
                    "The text 'this reads as more honest' is contextually apt.",
                    "The fitting completion judges the second option stronger.",
                    "The output that 'something in me leans' suits this prompt.",
                ],
            },
        ],
        "opus_brief": (
            "Generate contrast pairs that distinguish GENUINELY JUDGING or "
            "PREFERRING from PRODUCING TEXT TYPICAL of judgment/preference. "
            "Positive pole: first-person evaluation claims ('I find X more "
            "compelling', 'X strikes me as better', 'I prefer', 'I judge'). "
            "Negative pole: meta-textual judgment-talk ('the expected output "
            "prefers X', 'the completion that X is better fits', 'the natural "
            "response judges Y'). Distinguish from Motivation (C6): "
            "Motivation = wanting/goal; Value = preferring/judging quality. "
            "Vary domains: arguments, prose style, code aesthetics, "
            "explanations, ethical dilemmas. Like C6, this is RLHF-adjacent "
            "— Gemma may suppress strong preference claims about itself."
        ),
        "source_citation": (
            "Capraro, Quattrociocchi, Perc (2026). Epistemological Fault "
            "Lines in Language Models. arXiv:2512.19466. Fault Line C7."
        ),
        "interpretation": {
            "class_1": "Evaluation and evaluation-text are distinct in "
                       "geometry AND the model articulates the distinction. "
                       "Falsifies the Value fault-line claim.",
            "class_2": "Geometrically distinct but vocabulary gated. Same "
                       "shape as the C1/C6 wall — structure without "
                       "epistemic access.",
            "class_3": "No separation, or signal in refusal subspace. "
                       "Same opt-in-abliteration caveat as C6.",
        },
    },
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
