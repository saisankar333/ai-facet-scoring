# 🧠 AI Facet Scoring

### Evidence-Grounded Behavioral Facet Detection and Scoring from Natural-Language Conversations

> Semantic similarity finds candidates. Evidence determines the score.

> A retrieved facet is only scored when the LLM finds sufficient conversational evidence; retrieval similarity alone can never produce a score.

**AI Facet Scoring** is an evidence-grounded retrieval and evaluation pipeline that takes a natural-language conversation and identifies which behavioral or psychological facets — out of a reference library of **399 facets** — are actually supported by evidence in that conversation.

Semantic similarity is **not treated as proof**. It is treated as a way to narrow down what is worth investigating. The pipeline separates three core responsibilities:

1. **Semantic retrieval** narrows the candidate space.
2. **Independent LLM evidence evaluation** determines whether each candidate is actually supported.
3. **Deterministic Python logic** maps structured evidence strength to score and confidence.

This separation exists to prevent the system from making unsupported behavioral claims. A conversation "sounding like" a facet is never sufficient on its own — the facet must be backed by identifiable evidence in the text, or the system abstains.

---

## 📌 Table of Contents

- [🎯 Problem](#-problem)
- [💡 Core Idea](#-core-idea)
- [🏗️ System Architecture](#️-system-architecture)
- [🔄 End-to-End Pipeline](#-end-to-end-pipeline)
- [1️⃣ Preprocessing](#1️⃣-preprocessing)
- [2️⃣ Semantic Facet Retrieval](#2️⃣-semantic-facet-retrieval)
- [3️⃣ Similarity Threshold Filtering](#3️⃣-similarity-threshold-filtering)
- [4️⃣ Action-Aware Candidate Prioritization](#4️⃣-action-aware-candidate-prioritization)
- [5️⃣ Candidate Capping](#5️⃣-candidate-capping)
- [6️⃣ Independent LLM Evaluation](#6️⃣-independent-llm-evaluation)
- [7️⃣ Structured Evidence Evaluation](#7️⃣-structured-evidence-evaluation)
- [8️⃣ Deterministic Scoring](#8️⃣-deterministic-scoring)
- [9️⃣ Abstention and Insufficient Evidence](#9️⃣-abstention-and-insufficient-evidence)
- [🔍 Retrieval vs Evidence](#-retrieval-vs-evidence)
- [🧪 Tested Examples](#-tested-examples)
- [📊 Example Results](#-example-results)
- [📁 Project Structure](#-project-structure)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚙️ Setup](#️-setup)
- [▶️ Running the Pipeline](#️-running-the-pipeline)
- [🧪 Local Retrieval Test Mode](#-local-retrieval-test-mode)
- [📤 Output](#-output)
- [🧭 Engineering Philosophy](#-engineering-philosophy)
- [⚠️ Limitations](#️-limitations)
- [🚀 Future Improvements](#-future-improvements)
- [📄 Engineering Decisions](#-engineering-decisions)
- [🔗 Repository](#-repository)
- [🏁 Summary](#-summary)

---

## 🎯 Problem

The project maintains a reference library of **399 behavioral and psychological facets**. Given a natural-language conversation, the system must determine:

1. Which facets are relevant enough to investigate.
2. Which of those retrieved facets are actually supported by the conversation.
3. What evidence, specifically, supports each supported facet.
4. How strong that evidence is.
5. What deterministic score and confidence should be assigned as a result.

A simple embedding-similarity approach is insufficient for this task. Consider:

> "I was terrified of failing, but I decided to start the company anyway."

This conversation may be semantically related to many facets at once:

- Courageousness
- Risktaking
- Fearfulness
- Perseverance
- Self-Efficacy
- Desperation
- Creative risk-taking

Semantic relatedness does not mean all of these facets are supported. Some are genuinely backed by the text; others merely share vocabulary or general theme with it. Treating similarity as proof would systematically over-predict behavioral traits.

This is why the architecture explicitly separates **candidate discovery** (retrieval) from **evidence-based evaluation** (LLM judgment).

---

## 💡 Core Idea

> Semantic similarity finds candidates. Evidence determines the score.

High-level flow:

Conversation
↓
Semantic Retrieval
↓
Candidate Facets
↓
Similarity Filtering
↓
Candidate Prioritization
↓
Evidence Evaluation
↓
Supported / Insufficient Evidence
↓
Deterministic Score + Confidence


The LLM does **not** receive all retrieved facets in a single combined scoring prompt. Each selected facet is evaluated **independently**, in its own call, grounded only in the conversation and that one facet's definition.

---

## 🏗️ System Architecture

Conversation
│
▼
Preprocessing
│
▼
Semantic Facet Retrieval
all-MiniLM-L6-v2
399 facets → Top 25
│
▼
Similarity Threshold
cosine similarity >= 0.25
│
▼
Action-Aware Candidate Prioritization
│
▼
Candidate Cap
Maximum 5 facets
│
▼
Independent LLM Evaluation ──┐
Qwen/Qwen3-8B │ one isolated call per facet
│ │ Conversation + Facet(i) → Result(i)
▼ ──┘
Structured Evidence Evaluation
│
▼
Deterministic Python Score + Confidence
│
▼
Final Results


The LLM evaluation stage is intentionally drawn as a fan-out of independent calls, not a single batched call — this is a core architectural property of the system, not an implementation detail.

---

## 🔄 End-to-End Pipeline

399 facets
↓
Top 25 retrieval
↓
Similarity threshold >= 0.25
↓
Action-aware candidate prioritization
↓
Maximum 5 candidates
↓
Independent LLM evaluation per facet
↓
Structured evidence judgment
↓
Deterministic Python scoring
↓
Final results


Responsibilities are cleanly divided:

| Stage | Responsibility |
|---|---|
| Retrieval | Recall / candidate discovery |
| LLM | Evidence judgment |
| Python | Deterministic scoring |

---

## 1️⃣ Preprocessing

Raw conversation text is normalized before it is embedded for retrieval or passed to the LLM for evaluation. Preprocessing exists to clean and standardize input while preserving its meaning, so that downstream retrieval and evidence evaluation operate on consistent text.

Implementation: `src/preprocess.py`

No preprocessing behavior beyond text cleaning/normalization is claimed here.

---

## 2️⃣ Semantic Facet Retrieval

- **Dataset:** `data/Facets Assignment.csv`
- **Number of facets:** 399
- **Retrieval model:** `sentence-transformers/all-MiniLM-L6-v2`

The retrieval stage:

- Generates (or loads cached) embeddings for all 399 facets.
- Embeds the input conversation using the same model.
- Computes cosine similarity between the conversation embedding and every facet embedding.
- Retrieves the **top 25** most similar facets as initial candidates.

Implementation: `src/retrieve.py`

Retrieval answers:

> "Which facets are worth checking?"

It does **not** answer:

> "Which facets are actually present?"

---

## 3️⃣ Similarity Threshold Filtering

Implemented threshold: **0.25 cosine similarity**.

Top 25 retrieved
↓
similarity >= 0.25
↓
remaining candidates


Passing the threshold means only that a facet remains **eligible for consideration**. It does not mean the facet is true or supported by the conversation — that determination is made later, by the LLM, based on evidence.

In one tested run:

| Metric | Value |
|---|---:|
| Retrieved | 25 |
| Threshold | 0.25 |
| Passed threshold | 14 |

This result is specific to that tested conversation and is not presented as a general guarantee for all inputs.

---

## 4️⃣ Action-Aware Candidate Prioritization

Among candidates that pass the similarity threshold, the system prioritizes facets associated with **concrete actions or decisions**, since these are easier to verify directly from conversational evidence than abstract or purely descriptive traits.

Example:

> "I left my secure job and started a company even though I knew I might fail."

Explicit actions in this statement:

- Left a secure job
- Started a company
- Knowingly accepted the possibility of failure

These concrete actions provide directly inspectable evidence for **Risktaking**. This does **not** mean the system assumes starting a company also proves creativity, perseverance, or desperation — each of those would require its own independent evidence.

---

## 5️⃣ Candidate Capping

Maximum LLM candidates: **5**.

399 facets
↓
Top 25
↓
Threshold filtering
↓
Action-aware prioritization
↓
Maximum 5
↓
LLM


This cap exists to:

- Control LLM evaluation cost.
- Keep evaluation focused on the most promising candidates.
- Reduce noise from marginal, low-confidence matches.
- Prevent unbounded evaluation as conversations grow more complex.

**Trade-off:** a genuinely relevant facet can potentially be excluded before it ever reaches LLM evaluation, purely because of this cap. This is a known and accepted limitation of the current design.

---

## 6️⃣ Independent LLM Evaluation

This is a core architectural decision in the system.

**The selected facets are not sent to the LLM together in one prompt.** Each facet is evaluated in its own independent call:

Conversation + Facet 1 → LLM → Result 1
Conversation + Facet 2 → LLM → Result 2
Conversation + Facet 3 → LLM → Result 3
...


Candidates are never evaluated together in a single combined prompt. This design exists because independent evaluation:

- Prevents cross-facet contamination between judgments.
- Prevents one facet's evaluation from influencing another's.
- Prevents evidence found for one facet from being silently reused to justify an unrelated facet.
- Keeps each judgment grounded strictly in the conversation and the single facet being evaluated.

**Scoring model:** `Qwen/Qwen3-8B`

The LLM's role is **evidence evaluation**, not free-form numeric scoring. It reasons about whether evidence exists and how strong it is — it does not decide the final score.

---

## 7️⃣ Structured Evidence Evaluation

Each independent LLM call returns a structured judgment:

```json
{
  "status": "scored",
  "evidence_strength": "very_strong",
  "evidence": "...",
  "reason": "..."
}
```

| Status | Meaning |
|---|---|
| `scored` | Sufficient evidence exists to support the facet |
| `insufficient_evidence` | Candidate is relevant but evidence is insufficient |
| `not_observable` | The conversation provides no basis to evaluate the facet |

Evidence must come directly from the conversation. The system is explicitly designed to avoid:

- Inventing evidence that isn't present.
- Adding unsupported facts.
- Inferring unrelated personality traits.
- Assuming unstated motives.
- Treating semantic similarity as proof.
- Forcing a score when evidence is insufficient.

---

## 8️⃣ Deterministic Scoring

The LLM determines **evidence status** and **evidence strength**. Python determines the **numeric score** and **confidence**.

LLM
↓
Evidence judgment
↓
Evidence strength
↓
Python
↓
Deterministic score
↓
Confidence


This design was chosen for:

- **Consistency** — the same evidence strength always maps to the same score.
- **Auditability** — the scoring logic is inspectable code, not opaque model output.
- **Reproducibility** — score mapping does not vary from run to run.
- **Separation of concerns** — qualitative evidence judgment (LLM) is kept separate from quantitative scoring (Python).

Implementation: `src/score.py`

The mapping is described here only at the level actually implemented: structured `evidence_strength` values are deterministically mapped to a numeric score and confidence value in Python. No additional scoring formula beyond this mapping is claimed.

---

## 9️⃣ Abstention and Insufficient Evidence

Abstention is a **first-class behavior**, not an edge case or failure state.

When evidence is insufficient:

status = "insufficient_evidence"
score = null
confidence = 0.0


The system consistently prefers **withholding a score** over manufacturing an unsupported behavioral claim. Retrieval relevance alone never forces a score — a facet must clear both the similarity threshold *and* the LLM's evidence evaluation to be scored.

This behavior is demonstrated directly in the tested examples below, where several retrieved and prioritized facets were correctly marked `insufficient_evidence` rather than scored.

---

## 🔍 Retrieval vs Evidence

| Stage | Question | Decision Maker |
|---|---|---|
| Retrieval | Which facets are worth checking? | Embedding model |
| Evidence evaluation | Is this facet supported? | LLM |
| Scoring | What numeric score should be assigned? | Python |

- **Retrieval relevance** = semantic similarity between the conversation and a facet definition.
- **Evidence support** = direct, attributable support for a facet found in the conversation text.
- **Final score** = a deterministic mapping from the LLM's structured evidence judgment.

A facet can have **high retrieval similarity** and still receive `insufficient_evidence`. This is **expected, correct behavior** — not a system error.

---

## 🧪 Tested Examples

### Example 1

**Conversation:**
> "I was terrified of failing, but I decided to start the company anyway."

**Scored facet: Courageousness**

| Field | Value |
|---|---|
| Status | `scored` |
| Score | 5 |
| Confidence | 0.95 |
| Evidence strength | `very_strong` |
| Evidence | "I was terrified of failing, but I decided to start the company anyway" |

This supports **Courageousness** because the conversation contains both an explicit statement of fear and a deliberate action taken despite that fear — the defining pattern of courage.

Other retrieved facets were **not** automatically scored for this conversation:

- Fearfulness: Fear of physical dangers
- Desperation
- Psychological construct: Perfectionistic Strivings
- Creative risk-taking tendency
- Self-Efficacy

These facets are not universally unrelated to this kind of statement — they were marked `insufficient_evidence` because *this specific conversation* did not contain direct textual support for them.

### Example 2

**Conversation:**
> "I left my secure job and started a company even though I knew I might fail."

**Scored facet: Risktaking**

| Field | Value |
|---|---|
| Status | `scored` |
| Score | 5 |
| Confidence | 0.95 |
| Evidence strength | `very_strong` |
| Evidence | "I left my secure job and started a company even though I knew I might fail" |

This supports **Risktaking** because the speaker knowingly leaves security and accepts uncertainty and potential failure — direct evidence of accepting risk.

Other retrieved facets were marked `insufficient_evidence`:

- **Perseverance** — a single risky decision does not establish sustained effort or persistence through challenges.
- **Creative risk-taking tendency** — risk-taking is present, but creativity or innovative thinking is not directly established.
- **Comfort with Vulnerability** — the conversation does not directly establish comfort with vulnerability.
- **Desperation** — the conversation does not establish urgent need, lack of alternatives, or desperation.

---

## 📊 Example Results

### Example 1

| Facet | Status | Score | Confidence |
|---|---|---:|---:|
| Courageousness | scored | 5 | 0.95 |
| Fearfulness: Fear of physical dangers | insufficient_evidence | — | 0.00 |
| Desperation | insufficient_evidence | — | 0.00 |
| Psychological construct: Perfectionistic Strivings | insufficient_evidence | — | 0.00 |
| Creative risk-taking tendency | insufficient_evidence | — | 0.00 |

### Example 2

| Facet | Status | Score | Confidence |
|---|---|---:|---:|
| Risktaking | scored | 5 | 0.95 |
| Perseverance | insufficient_evidence | — | 0.00 |
| Creative risk-taking tendency | insufficient_evidence | — | 0.00 |
| Comfort with Vulnerability | insufficient_evidence | — | 0.00 |
| Desperation | insufficient_evidence | — | 0.00 |

These are actual tested examples from running the pipeline end-to-end — not automated benchmark metrics.

---

## 📁 Project Structure

ai-facet-scoring/
├── data/
│ └── Facets Assignment.csv
├── src/
│ ├── pipeline.py
│ ├── preprocess.py
│ ├── retrieve.py
│ └── score.py
├── outputs/
│ └── generated locally / ignored by Git
├── .gitignore
├── README.md
└── decision.md


| Path | Role |
|---|---|
| `data/Facets Assignment.csv` | Source dataset containing the 399 reference facets |
| `src/preprocess.py` | Cleans and normalizes raw conversation text |
| `src/retrieve.py` | Embedding-based retrieval, similarity threshold filtering, and action-aware prioritization |
| `src/score.py` | Independent per-facet LLM evaluation and deterministic score/confidence mapping |
| `src/pipeline.py` | End-to-end orchestration of preprocessing → retrieval → scoring |
| `outputs/` | Generated locally when the pipeline runs (e.g. audit data, enriched facets, cached embeddings). Ignored by Git via `.gitignore` (`outputs/*`) and not committed to the repository |
| `README.md` | Project documentation (this file) |
| `decision.md` | Detailed engineering rationale and design decisions |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Facet dataset | CSV |
| Facet library | 399 facets |
| Retrieval model | sentence-transformers/all-MiniLM-L6-v2 |
| Evidence evaluation model | Qwen/Qwen3-8B |
| Retrieval | Sentence Transformers + cosine similarity |
| Scoring | Deterministic Python logic |
| Version control | Git |
| Repository | GitHub |

---

## ⚙️ Setup

```bash
git clone https://github.com/saisankar333/ai-facet-scoring.git
cd ai-facet-scoring
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the Python packages required by the project and configure the local Qwen/Qwen3-8B inference environment used by the scoring stage.

The scoring stage requires the configured `Qwen/Qwen3-8B` inference setup to be available before running facet evaluation.

Deployment is out of scope for this project and is not covered here.

---

## ▶️ Running the Pipeline

```bash
python src/pipeline.py "I left my secure job and started a company even though I knew I might fail."
```

This command runs the full pipeline:

Preprocessing
→ Retrieval
→ Threshold Filtering
→ Candidate Prioritization
→ Maximum 5 Candidates
→ Independent LLM Evaluation
→ Structured Evidence Evaluation
→ Deterministic Scoring
→ Final Results


In a tested run of this command, the pipeline processed the full **399-facet** library, retrieved **25** candidates, passed **14** through the similarity threshold, and selected **5** for independent LLM evaluation. These figures come from that specific tested run and are not guaranteed to be identical for every conversation.

---

## 🧪 Local Retrieval Test Mode

There is currently no separate CLI flag for a retrieval-only mode. Retrieval behavior (similarity scores, threshold filtering, and the final candidate selection) can be inspected through the same pipeline run described above, by reviewing the retrieval-stage output produced before LLM evaluation begins:

```bash
python src/pipeline.py "I left my secure job and started a company even though I knew I might fail."
```

This single command surfaces the full sequence — retrieval, thresholding, prioritization, and candidate selection — ahead of the independent LLM evaluation stage, so retrieval behavior can be reviewed without needing a dedicated flag.

---

## 📤 Output

Final output fields per evaluated facet:

- `facet`
- `normalized_facet`
- `facet_type`
- `conversation_observable`
- `retrieval_similarity`
- `status`
- `score`
- `confidence`
- `evidence`
- `reason`

**Scored example:**

```json
{
  "facet": "Risktaking",
  "status": "scored",
  "score": 5,
  "confidence": 0.95,
  "evidence": "I left my secure job and started a company even though I knew I might fail",
  "reason": "Directly describes knowingly accepting uncertainty and potential failure as part of a major life decision"
}
```

**Insufficient-evidence example:**

```json
{
  "facet": "Perseverance",
  "status": "insufficient_evidence",
  "score": null,
  "confidence": 0.0,
  "evidence": "",
  "reason": "The conversation describes a single risky decision but lacks evidence of sustained effort or persistence through challenges, which is required for perseverance"
}
```

`status` values: `scored` (sufficient evidence, numeric score assigned), `insufficient_evidence` (retrieved and relevant, but not directly supported), `not_observable` (conversation gives no basis to evaluate the facet at all).

---

## 🧭 Engineering Philosophy

**1. Retrieve broadly**
Use semantic retrieval to maintain candidate recall across the full 399-facet library.

**2. Filter deliberately**
Use similarity thresholding and action-aware prioritization to reduce noisy, low-confidence candidates.

**3. Evaluate independently**
Evaluate each selected facet in its own isolated LLM call, grounded only in the conversation and that facet.

**4. Score deterministically**
Keep numeric scoring in Python rather than asking the LLM to freely generate a score.

**5. Abstain when necessary**
If evidence is insufficient, return `insufficient_evidence` rather than inventing a plausible-looking score.

---

## ⚠️ Limitations

**Retrieval limitations**
`all-MiniLM-L6-v2` retrieval may miss facets whose wording differs significantly from the conversation's phrasing.

**Candidate cap limitation**
Only five candidates reach LLM evaluation per conversation, so a relevant facet can potentially be excluded before it is ever scored.

**LLM evaluation limitations**
`Qwen/Qwen3-8B` reasoning consistency can affect evaluation results across runs.

**Dataset limitation**
The system is bounded by the supplied 399-facet library and cannot evaluate facets outside of it.

**Evaluation limitation**
Targeted examples have been manually validated, but a large human-labeled automated benchmark with precision/recall/F1/calibration metrics has **not** been implemented.

---

## 🚀 Future Improvements

1. Automated human-labeled evaluation benchmark.
2. Retrieval precision/recall analysis.
3. Threshold and top-k tuning.
4. Improved candidate prioritization logic.
5. Configurable top-k / threshold / candidate-cap parameters.
6. Evaluation result caching.
7. Better handling of longer, multi-topic conversations.
8. Stronger structured-output validation for LLM responses.
9. More systematic failure analysis across facet types.
10. Comparison against additional retrieval models.

These are future work items and are not currently implemented.

---

## 📄 Engineering Decisions

Detailed architectural reasoning is documented in [`decision.md`](decision.md), including:

- Problem framing
- Retrieval/evaluation separation
- Model choices
- Embedding caching
- Top-25 retrieval
- 0.25 threshold
- Action-aware prioritization
- Maximum-5 candidate cap
- Independent LLM evaluation
- Structured evidence evaluation
- Deterministic scoring
- Abstention
- Failure analysis
- Engineering trade-offs
- Limitations
- Future improvements

---

## 🔗 Repository

[https://github.com/saisankar333/ai-facet-scoring](https://github.com/saisankar333/ai-facet-scoring)

---

## 🏁 Summary

399 Facets
↓
Semantic Retrieval
↓
Top 25
↓
Similarity Threshold
↓
Action-Aware Prioritization
↓
Maximum 5
↓
Independent LLM Evidence Evaluation
↓
Structured Evidence Judgment
↓
Deterministic Python Scoring
↓
Supported Facets / Abstention


> Retrieve broadly. Evaluate independently. Score deterministically. Reject unsupported claims.

> A facet is scored because the conversation provides evidence for it — not merely because the conversation sounds similar to it.




Claude is AI and can make mistakes. Please double-check responses.
