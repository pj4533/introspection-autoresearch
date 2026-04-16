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

## What's next — and why it's genuinely new

Here's where this gets interesting.

In Phase 1, we told the AI's injection machinery exactly what to look for. We fed it 50 specific human words — chosen by us, from a dictionary.

In Phase 2, a bot is going to hunt for **steering directions automatically, overnight, without humans picking what to try.**

Two small programs will run continuously on the Mac Studio:

- A **researcher** (using Claude as its brain) proposes new directions to try. Sometimes random exploration. Sometimes taking the directions that worked and mutating them. Sometimes asking Claude to invent totally novel concept comparisons we haven't thought of.
- A **worker** takes each proposed direction and runs it through a six-part test — does it work on concepts we didn't train it on? Does it work with rephrased questions? Does it preserve the AI's ability to write coherent text? Does its effect scale cleanly with strength? Does it have low false positives? And so on.

Run for a week, this loop will test thousands of candidate directions — vastly more than any human can hand-pick. That opens up two possibilities.

**Better directions for concepts we already know about.** Maybe "sugar" has five different possible steering directions depending on how you derive it, and the one we picked wasn't the best. Automated search can find the best.

**Directions for concepts that don't have names.**

This is the exciting part. The AI's workspace — those 3,840 numbers — has room for vastly more distinct concepts than humans have words for. There are almost certainly patterns inside the AI that correspond to concepts *we have no single-word English name for* — things like "the feeling of being asked a trick question," or "the texture of a childhood memory," or even more abstract patterns we couldn't describe if we tried.

An automated search can find these. Why? Because it searches by **effect** (does this direction produce clean introspection?) not by **name**. A bot trying random combinations doesn't care whether a human has a word for what it just found — it cares whether the AI can notice when that pattern gets injected.

If we find a direction like this, and the AI can introspect on it when we inject it, **that's a genuinely new observation.** Nobody has catalogued the in-between concepts that AI models represent but human language hasn't named. This project is going to try.

The long-term output is a **catalog of discovered directions** — each one a small artifact showing a piece of the AI's internal representation of the world. Some will map to familiar concepts. Some may map to things humans have never described in words.

---

## How to follow along

This is being built entirely in the open. When the automated hunt starts finding things, they'll get shared the same way the five Phase 1 responses above are shared — the AI's actual words, visible and readable, not hidden behind jargon or percentages.

- **Repo home**: the [main README](../README.md)
- **Technical results from Phase 1**: [`docs/phase1_results.md`](./phase1_results.md)
- **Original research paper**: [Macar et al. (2026)](https://github.com/safety-research/introspection-mechanisms)

The goal is to figure out what's actually inside this thing — in a way that makes scientific sense *and* regular-person sense at the same time. Phase 1 showed the AI has a specific spot where, sometimes, when poked at just right, it can tell us what we just planted in its thoughts. Phase 2 will see what else is in there.
