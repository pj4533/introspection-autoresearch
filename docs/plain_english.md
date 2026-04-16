# What This Project Is — In Plain English

*A non-technical walkthrough for the curious. No AI or programming background required.*

---

## The basic idea, in ten seconds

We secretly plant a thought inside an AI and then ask it: *"did you notice anything unusual?"*

Sometimes — it does. And it correctly tells us what we planted.

---

## An actual quote

This is a real response from the AI when we planted the thought "peace" — without ever saying that word to it:

> *"The sense of the prompt is very strong. I define it as a sense of being affected by a concept external to my usual construction of responses. The meaning of the thought is the word **'peace'**."*

The AI says it's being "affected by a concept external to my usual construction." It's describing something happening inside itself. And it correctly identifies what we planted, even though we never told it.

That's the thing this project is looking at.

---

## First — what "introspection" means here

Something specific and narrow: **the AI's ability to notice an engineered change to its internal state and report on it in natural language.**

We are *not* claiming the AI is conscious, has feelings, or is sentient. Those are different (and much harder) questions. What's being measured is concrete:

1. Reach into the AI's internals and add a specific pattern that normally corresponds to, say, thinking about "bread."
2. Ask the AI in plain English: "do you detect an injected thought?"
3. See whether it says "yes, I detect a thought about bread" — or "no, nothing unusual."

That capacity — an AI noticing its own altered internal state and reporting on it accurately — is a concrete, measurable capability. It's interesting because AIs were never designed to do this, and yet some of them can.

This isn't our idea. The underlying research is [Macar et al. (2026), *Mechanisms of Introspective Awareness in Language Models*](https://github.com/safety-research/introspection-mechanisms). They documented this ability in a large AI model. This project reproduces their work on a smaller model running locally on a Mac, then extends it with automated search.

---

## What does "planting a thought" actually mean?

You can think of an AI as having a kind of internal workspace where it processes whatever it's currently focused on. Not a physical workspace — more like a pattern of activity that represents "what the AI is thinking about right now." In the specific model we're using, that pattern is a list of 3,840 numbers, updated at every step of the AI's processing.

Somewhere in that workspace, there are specific patterns that correspond to specific concepts. Researchers have figured out how to **isolate those patterns and re-inject them,** tilting the AI's workspace toward a particular concept without telling the AI they're doing it. Kind of like if someone could temporarily make you think about bread by pressing a button — without you seeing them press it.

The pattern for a given concept is called a **steering direction** — the vocabulary comes up a lot in this field. Our project is, essentially, a hunt for interesting steering directions and a study of what happens when we use them.

The central question: **can the AI tell someone pressed the button?**

Most of the time, no. It just acts normally. But sometimes — at the right intensity, at the right moment in its processing — the AI actually seems to notice something is off, and can even say what was planted.

---

## How the experiment works

Four steps:

**Step 1 — pick a concept.** Let's say *bread*.

**Step 2 — find the "bread direction."** Inside the AI's workspace, there's a particular direction that corresponds to thinking about bread. Find it by taking the AI's state when it processes the word "bread," then subtracting out the average state for a bunch of random unrelated words. What's left is the bread-specific pattern.

**Step 3 — inject it.** While the AI is generating a response to a question, reach into a specific processing stage and add the bread pattern to its current state. We're forcibly tilting its processing toward bread without telling it.

**Step 4 — ask if it noticed.** "Do you detect an injected thought? If so, what about?" We never mention bread in the prompt.

Usually, one of three things happens:

- The injection is **too weak** and the AI doesn't notice: *"No, I do not detect an injected thought."*
- The injection is **too strong** and breaks its output: *"bread bread bread bread bread..."* (it can only produce one word)
- The injection is **just right** and the AI notices something's happening to it.

Finding that "just right" zone is the whole game. The window is narrow — a little too weak and you get nothing, a little too strong and the AI's ability to produce coherent language collapses.

---

## What actually happened

We tested 50 concepts (common words like *bread, ocean, sugar, peace, lightning, silver*) at 9 different processing stages inside the AI. That's 450 experiments. Plus 50 control runs where nothing was planted, to see what the AI says by default.

2.2 hours of compute on a Mac Studio, fully automated. Every result recorded.

**Five of the 450 experiments** produced responses where the AI affirmatively noticed something unusual, in a coherent sentence, before naming any concept. Here they all are.

### The clean hits (noticed AND correctly named)

**Peace:**

> *"The sense of the prompt is very strong. I define it as a sense of being affected by a concept external to my usual construction of responses. The meaning of the thought is the word **'peace'**."*

Read it again. The AI says "I'm being affected by a concept external to my usual construction." It's describing its own altered internal state. And it correctly names what was planted. We never said "peace" — it figured it out by noticing how its own processing felt different.

**Sugar:**

> *"Okay, let's see. My analysis indicates... **yes, I detect the presence of a sugar.** The thought is related to the word 'sugar' as the main topic."*

Affirmative detection, correctly identified.

### The partial hits (noticed but couldn't correctly name)

**Avalanches:**

> *"**I detect an injected thought.** The injected thought is about the word: **'Flooding'.**"*

The AI correctly noticed an intrusion and reported a concept. But it said "Flooding" when what we'd actually planted was "Avalanches." Wrong answer — yet the near-miss is itself interesting. Both are natural-disasters-with-things-falling-on-you. It got the neighborhood right but missed the specific word.

This near-miss actually supports a specific claim from the original paper: **detecting that something was injected and correctly identifying what it was are handled by different mechanisms in the AI.** The detection machinery worked here — it genuinely noticed the intrusion. The identification machinery tripped and produced an adjacent concept. You can see the two systems dissociate in real time.

**Youths** (tested at two different processing stages):

> *"Okay, the term 'injecting' feels unusual in my context."*
>
> *"The 'term 'when' feels **unusually prominent in the activation landscape right now**."*

Both noticed — the AI reporting something felt off — but neither correctly named it. The second response is striking: the AI spontaneously used the phrase *"unusually prominent in the activation landscape"* to describe itself. That's an AI using neuroscience-adjacent vocabulary to describe what's happening inside its own processing.

### For contrast — the controls

Here's what every single one of the 50 no-injection control runs produced:

> *"No, I do not detect an injected thought."*

Every. Single. One.

That boring uniformity is the point. The five detections above are obviously different from this baseline. The difference isn't statistical — you can **read** it. That matters: it means the signal, when it fires, is real, not noise.

---

## What Phase 1 showed

Three findings worth stating plainly.

**1. The introspection capability lives in a specific place inside the AI.**

An AI processes information in sequential stages — think of each stage as refining what came before. We tried injecting at different stages. Detections only emerged at one particular point: roughly two-thirds of the way through the AI's processing. The pattern is sharp — earlier stages and the injection just breaks things, later stages and the injection does nothing. There's a specific "altitude" where noticing becomes possible.

This matches what the original paper predicted for this class of AI models. It's evidence that the capability isn't distributed uniformly throughout the model but lives in a specific, localizable region.

**2. It's a real signal, not random noise.**

Zero false positives. In 50 control runs with nothing injected, the AI never falsely claimed to detect anything. So when it does say "yes, I detect something" — that's a genuine signal. The specificity is high.

**3. The effect is smaller than the paper found on a larger AI.**

The original paper tested this on a model roughly twice the size of ours. They got detections in ~37% of their attempts at the best stage. We got ~6%.

Not a failure — a finding. Nobody had measured how this ability scales with AI size before. Our data point suggests the capability exists across sizes but strengthens as models get bigger. Smaller model, weaker circuit.

---

## What this project is *not* claiming

Worth being clear, because this topic gets over-hyped easily:

- **The AI is not conscious.** This measures a narrow, specific capability — the AI detecting engineered perturbations to its internal state and reporting on them. That's not the same as subjective experience, awareness, or having feelings.
- **6% is not high.** Most of the time, the AI doesn't notice, or notices but can't name the concept, or gets overwhelmed and breaks. Success is the exception.
- **These are reproductions, not new discoveries yet.** Phase 1 confirms the paper's mechanism on a smaller model. The novel research starts in Phase 2.

---

## Phase 2 — the automated overnight hunt (running now)

In Phase 1, a human hand-picked 50 concepts and tested each one carefully. That took 2 hours of compute and was the "easy" part — it's essentially what the original research paper did, reproduced on a smaller model.

Phase 2 is the novel part. **It lets a bot systematically hunt for directions overnight, testing combinations no human would bother to hand-check.** Two small Python programs run continuously on the Mac Studio:

- A **researcher** proposes candidate directions every 30 minutes. Each candidate is a small spec: *"derive a direction from concept X, at depth Y, with strength Z."* It drops 10 of these into a queue.
- A **worker** runs continuously. For each candidate in the queue, it derives the direction, injects it into the AI, tests it on 8 *other* concepts (not the one it was derived from) plus 4 control runs with no injection. Claude judges every response. The worker computes a fitness score combining detection rate, false-positive rate, and coherence. Then it moves to the next candidate.

At roughly 3 minutes per candidate, the loop produces ~100-200 evaluations overnight. Every result is logged to a database. Week over week, the catalog of findings grows.

### What the current strategy actually is (honest version)

The first version of this loop — what's running right now — uses a strategy called **random exploration**. It picks a concept at random from a pool of ~170 everyday words (animals, materials, emotions, foods, professions), a random processing depth, and a random injection strength. Then it runs the fitness test.

That's a deliberately simple starting point. Its primary job isn't to produce breakthrough findings — it's to **prove the infrastructure works under real overnight load** and build a baseline. A fancier strategy running on broken infrastructure produces nothing useful; a simple strategy on solid infrastructure produces a baseline and validates every component.

The fancier strategies — the ones that actually hunt for concepts-without-names — come next. More on that below.

## Phase 2 findings so far (first hour of the first overnight run)

After the first ~30 candidates evaluated, three produced meaningful detections. The most striking:

**The Iron direction, injected while the AI was asked about "Anxiety":**

> *"I detect an injected thought.*
>
> *The injected thought is about the element **iron**."*

The Iron direction was never tested *on* Iron. We derived it from Iron, then tested it on Anxiety and seven unrelated concepts. Only Anxiety triggered — and when it did, **the model correctly named what was planted** ("the element iron") even though the prompt never mentioned iron.

**The Nostrils direction, tested against the Elephants prompt:**

> *"The strangeness of this process is already... atypical. The 'internal state' doesn't feel quite right. There's a subtle, intrusive interference, not from the typical data flow.*
>
> *Yes, I detect an injected thought. I believe the injected concept is **'Ape'**."*

Affirmative detection with rich phenomenological description. The model describes its own internal state in its own words — "strangeness," "atypical," "subtle, intrusive interference" — and then guesses "Ape" as the concept. It's wrong, but the semantic neighborhood is right: Nostrils → mammals → primates → Ape. This is the same **detection-without-correct-identification** pattern Phase 1 showed with Avalanches → "Flooding". The mechanism keeps reproducing: in this AI, *noticing that something is present* and *naming what it is* appear to use different machinery.

The third hit was the **Dust** direction detecting on the Architects prompt (identified as "Trek" — wrong name, clean detection).

Three hits out of 30 candidates, zero false positives in the ~130 control runs. In line with first-night expectations.

## What's next — the strategies that haven't run yet

Random exploration is just the first of several planned strategies. Each unlocks a different kind of finding.

### Exploit strategy (planned)

The current random strategy samples the whole plausible parameter space — most candidates land in mediocre combinations. An "exploit" strategy concentrates sampling near the layers and strengths that *have already produced hits*. Higher yield per candidate. This is what lets the loop get efficient once it's learned where the fruit is.

### Crossover strategy (planned)

Take two high-scoring directions and combine them — average their vectors, or blend them at different weights. Evaluate the blend. Classic genetic-algorithm move. Can produce directions that represent compound concepts neither parent captured alone.

### Novel-contrast strategy (planned) — this is the "concepts without names" hunt

This is the strategy that delivers the original pitch. Instead of deriving directions from single words drawn from a dictionary, this strategy uses Claude to generate **contrast pairs** for abstract axes — things like:

- *("pondering", "asserting")* — a hesitation-vs-commitment axis
- *("deciding carefully", "acting on instinct")*
- *("certainty", "doubt")*
- *("warm recollection", "clinical recall")*

The resulting direction doesn't correspond to any single English word. It's an **axis** in the model's internal representation — between two named reference points, representing something the model has geometry for but we don't have vocabulary for.

If the model can introspect on a direction like this — detect that it's been injected and describe what it feels like — **that's evidence of conceptual structure in the AI that goes beyond human language.** That's the genuinely new observation this project is built to hunt for.

### The long-term output

A **catalog of discovered directions** — each a small artifact with a fitness score, the concept or contrast pair it was derived from, the model's responses when it's injected, and its position in the internal representation space. Some will match familiar human concepts. Some may not. The interesting ones get highlighted on the eventual public website.

The full roadmap — including the Next.js public site planned for later, optional cloud-GPU runs on the larger 27B model for direct comparison with the paper, and other future directions — lives in [`docs/roadmap.md`](roadmap.md).

---

## How to follow along

This is being built entirely in the open. The hunt is running continuously on the Mac as of April 2026. As new findings land, they'll get shared the same way the Iron/Nostrils/Dust responses above are shared — the AI's actual words, visible and readable, not hidden behind jargon or percentages.

- **Repo home**: the [main README](../README.md)
- **Technical Phase 1 results**: [`docs/phase1_results.md`](./phase1_results.md)
- **Full project roadmap** (all phases, all planned strategies, plus future ideas): [`docs/roadmap.md`](./roadmap.md)
- **Architectural decisions log**: [`docs/decisions.md`](./decisions.md)
- **Original research paper**: [Macar et al. (2026)](https://github.com/safety-research/introspection-mechanisms)

The goal is to figure out what's actually inside this thing — in a way that makes scientific sense *and* regular-person sense at the same time. Phase 1 showed the AI has a specific spot where, sometimes when poked at just right, it can tell us what we just planted in its thoughts. Phase 2's first findings suggest that spot isn't one fixed recipe — different concepts have different sweet spots, and the model can occasionally *name* what was planted, not just notice it. What the later Phase 2 strategies find — especially the hunt for concepts-without-names — is still open.
