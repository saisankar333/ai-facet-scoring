
markdown
# decision.md — Engineering Rationale for AI Facet Scoring

This document explains **why** AI Facet Scoring is built the way it is. The [README](README.md) describes *what* the system does; this file documents the reasoning, trade-offs, rejected alternatives, and accepted limitations behind each major engineering decision.

Repository: https://github.com/saisankar333/ai-facet-scoring

---

## 1. Problem Definition

Given a natural-language conversation and a reference library of **399 behavioral/psychological facets**, the system must determine:

1. Which facets are worth investigating.
2. Which of those facets are actually supported by the conversation.
3. What specific evidence supports each supported facet.
4. How strong that evidence is.
5. What deterministic score and confidence should result.

Semantic similarity alone is insufficient to answer these questions. Consider the real tested example:

> "I left my secure job and started a company even though I knew I might fail."

This statement directly supports **Risktaking** — it explicitly describes leaving security and accepting uncertainty. But it does **not** automatically establish **Perseverance**, **Creative risk-taking tendency**, **Comfort with Vulnerability**, or **Desperation**, even though all of these facets may be semantically related to the general theme of the conversation.

This is the core problem the architecture is built around:

**Semantic relevance ≠ Evidence support.**

A system that scores purely on similarity would systematically over-predict behavioral traits. The project is designed to avoid that failure mode.

---

## 2. Core Engineering Principle

> Semantic similarity finds candidates. Evidence determines the score.

The system separates three distinct responsibilities:

- **Retrieval** → candidate discovery
- **LLM** → evidence judgment
- **Python** → deterministic scoring

Retrieval similarity can never directly produce a score. A facet must pass through retrieval, filtering, prioritization, and — most importantly — independent LLM evidence evaluation before any numeric score is assigned.

---

## 3. Decision: Separate Retrieval from Evidence Evaluation

Retrieval and evidence evaluation are treated as separate responsibilities with separate questions:

- **Retrieval** answers: *"Which facets are worth checking?"*
- **LLM evaluation** answers: *"Is this particular facet actually supported by the conversation?"*

Treating embedding similarity as proof carries real risks:

- **Semantic false positives** — a facet can rank highly without being true.
- **Shared vocabulary** — conversations and facet definitions can overlap in wording without overlapping in meaning.
- **Related but distinct concepts** — e.g., risk-taking and desperation can appear similar in embedding space while being behaviorally distinct.
- **Unsupported behavioral claims** — scoring on similarity alone risks asserting things about a person that the conversation never actually established.

Retrieval is treated strictly as **candidate generation**, not truth classification.

---

## 4. Decision: Retrieve Against All 399 Facets

Retrieval is performed against the complete 399-facet reference library rather than a manually curated subset.

Purpose:

- Enables broad candidate discovery across the full facet space.
- Avoids the risk of manually and arbitrarily restricting which facets can ever be considered.
- Preserves candidate recall as much as is practical at the retrieval stage.

This does not claim or guarantee perfect recall — retrieval quality is still bounded by the embedding model (see Section 5).

---

## 5. Decision: Retrieval Model

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

This model converts both the conversation and the facet definitions into embeddings so that semantic similarity can be computed via cosine similarity. It was chosen as a practical, widely used sentence-embedding model for this retrieval task.

This is **not** claimed to be the best possible embedding model for this problem. A known limitation is documented rather than hidden: conversations phrased very differently from a facet's definition wording may be under-retrieved, since the model relies on semantic proximity in embedding space rather than explicit conceptual reasoning.

---

## 6. Decision: Cached Facet Embeddings

Facet embeddings are cached and reused rather than recomputed on every run, since the 399-facet library is stable across runs.

Rationale:

- Avoids repeated embedding computation for a dataset that does not change between runs.
- Improves efficiency across repeated pipeline executions.
- Separates fixed facet preprocessing (embedding the stable facet library) from per-conversation processing (embedding the input conversation).

No specific measured speedup numbers are claimed here — only the architectural rationale for caching.

---

## 7. Decision: Top-25 Retrieval

The retrieval stage returns the **top 25** facets by cosine similarity.

Rationale:

- Initial retrieval is intentionally kept broad enough to support candidate discovery, rather than immediately narrowing to a final set.
- Later stages (threshold filtering, action-aware prioritization, candidate capping) perform the stricter narrowing.
- Retrieval alone should not determine the final five facets sent to the LLM — that responsibility is spread across multiple, more targeted stages.

**Trade-off:** a larger top-k value surfaces more candidates and preserves more recall, but also increases noise and downstream filtering/evaluation work. 25 is the value used in the current implementation; it is not claimed to be mathematically or empirically optimal.

---

## 8. Decision: Similarity Threshold of 0.25

**Threshold:** cosine similarity `>= 0.25`

This threshold removes clearly weak candidates before they reach the more expensive LLM evaluation stage.

**Important distinction:** passing the 0.25 threshold does **not** mean a facet is true. It only means the facet remains **eligible for further evaluation** — the actual determination of support still happens at the LLM evidence-evaluation stage.

From one tested run:

| Metric | Value |
|---|---:|
| Total facets | 399 |
| Retrieved | 25 |
| Passed threshold (0.25) | 14 |
| Selected for LLM evaluation | 5 |

These numbers come from a specific tested run of the pipeline. They are not benchmark metrics and are not a universal guarantee for every conversation. 0.25 is not claimed to be mathematically or empirically optimal.

---

## 9. Decision: Action-Aware Candidate Prioritization

Among candidates that pass the similarity threshold, the retrieval pipeline applies candidate-selection/prioritization logic that favors facets connected to **concrete actions or decisions** in the conversation, because concrete actions are easier to verify directly from conversational evidence than abstract or purely descriptive traits.

This is implemented as prioritization logic within `src/retrieve.py`, not as a separate machine-learning classification model. No dedicated ML model is used to detect or classify actions; the prioritization is part of the existing retrieval/filtering pipeline.

Example:

> "I left my secure job and started a company even though I knew I might fail."

Observable actions in this statement:

- Left a secure job
- Started a company
- Knowingly accepted possible failure

These concrete actions provide strong, directly inspectable support for **Risktaking**. They do **not** automatically prove:

- Creativity
- Perseverance
- Desperation
- Comfort with vulnerability

Each of those facets still requires its own independent LLM evidence evaluation before it can be scored — action-aware prioritization only affects which candidates are selected for that evaluation, not whether they are ultimately supported.

---

## 10. Decision: Maximum 5 Candidates

The number of candidates that reach LLM evaluation is capped at **5**.

Rationale:

- LLM evaluation is comparatively more expensive than embedding-based retrieval.
- A cap reduces the number of noisy, marginal candidates reaching evaluation.
- It keeps evaluation focused on the most promising candidates.
- It controls overall evaluation cost per conversation.

**Accepted trade-off:** a genuinely relevant facet can be excluded before it ever reaches LLM evaluation if it does not survive retrieval, thresholding, and prioritization. This is a known and deliberately accepted limitation, not an oversight. The value of 5 is not claimed to be empirically optimal — it reflects the current implementation's balance between coverage and cost.

---

## 11. Decision: Independent LLM Evaluation Per Facet

This is one of the most important architectural decisions in the project.

**The selected facets are not sent to the LLM together in one combined prompt.**

Instead, each candidate facet is evaluated in its own independent LLM call:
Conversation + Facet 1 → LLM → Result 1
Conversation + Facet 2 → LLM → Result 2
Conversation + Facet 3 → LLM → Result 3
...

Rationale:

- **Prevents cross-facet contamination** — one facet's evidence cannot bleed into another's judgment.
- **Prevents anchoring** — the model cannot let its judgment on one facet bias its judgment on the next.
- **Prevents evidence reuse** — evidence identified for one facet cannot be silently reused to justify an unrelated facet.
- **Keeps evidence attribution isolated** — each result is traceable to a single, isolated evaluation.
- **Improves auditability** — each facet's reasoning can be reviewed independently of every other facet's reasoning.

**Trade-off:** independent calls require more total LLM invocations than a single combined request would. This additional cost is deliberately accepted in exchange for evidence isolation and cleaner, more auditable evaluation. Independent evaluation does not guarantee perfect reasoning on every call — it reduces a specific, identifiable class of errors (cross-facet contamination), not all possible LLM reasoning errors.

---

## 12. Decision: Qwen/Qwen3-8B for Evidence Evaluation

**Model:** `Qwen/Qwen3-8B`

This model's role in the pipeline is strictly **qualitative evidence evaluation**. For each facet it is asked to evaluate, it determines:

- `status`
- `evidence_strength`
- `evidence`
- `reason`

The LLM does **not** determine the final numeric score — that is computed deterministically downstream (see Section 15). `Qwen/Qwen3-8B` was selected as the model used in the current implementation; it is not claimed to be the objectively best model available for this task.

---

## 13. Decision: Structured LLM Output

Each LLM evaluation call returns a structured judgment:

```json
{
  "status": "scored",
  "evidence_strength": "very_strong",
  "evidence": "...",
  "reason": "..."
}
```

Possible `status` values:

| Status | Meaning |
|---|---|
| `scored` | Sufficient evidence exists to support the facet |
| `insufficient_evidence` | The facet is relevant but not directly supported by the conversation |
| `not_observable` | The conversation provides no basis to evaluate the facet at all |

Structured output was chosen because it:

- Produces predictable, machine-parseable downstream handling.
- Cleanly separates qualitative evidence judgment (LLM) from quantitative numeric scoring (Python).
- Makes evaluations easier to audit — each field can be reviewed independently.
- Enables deterministic downstream mapping from evidence strength to score.

Structured parsing is handled in `src/score.py`; the system does not claim exhaustive schema validation.

---

## 14. Decision: Evidence-Constrained Evaluation

The evidence evaluation stage is intentionally constrained. The system is designed so that it should not:

- Invent evidence that is not present in the conversation.
- Add unsupported facts.
- Infer unrelated personality traits.
- Infer unstated motives.
- Treat semantic similarity as proof.
- Force a score when evidence is insufficient.

The evaluation question is deliberately framed as *"does the conversation support this facet?"* rather than *"does this facet seem plausible given the conversation?"* — the latter framing is exactly what would allow similarity and plausibility to masquerade as evidence. The system is designed to be conservative by default.

---

## 15. Decision: Deterministic Python Scoring
LLM
↓
Evidence status + evidence strength
↓
Python
↓
Numeric score + confidence

Numeric scoring is performed in Python rather than by the LLM, for several reasons:

- **Consistency** — the same evidence strength always maps to the same score.
- **Auditability** — the mapping logic is inspectable code, not opaque model behavior.
- **Reproducibility** — scores do not vary across runs due to LLM sampling variance.
- **Separation of concerns** — qualitative evidence judgment (LLM) is kept independent from quantitative score mapping (Python).

The LLM does not freely generate the final numeric score under this design.

Implementation: `src/score.py`

No scoring formula beyond what is actually implemented is claimed here — the description above is intentionally kept at the level of "structured evidence strength is deterministically mapped to score and confidence," matching the current implementation.

---

## 16. Decision: Abstention and Insufficient Evidence

Abstention is treated as a **first-class outcome**, not an edge case.

When evidence is insufficient:
status = "insufficient_evidence"
score = null
confidence = 0.0

Rationale:

- Avoids making unsupported behavioral claims.
- Prevents retrieval similarity from being converted into a forced score.
- Makes uncertainty explicit rather than hiding it behind a plausible-looking number.
- Is safer than manufacturing a score that merely looks reasonable.

From the tested example:

> "I left my secure job and started a company even though I knew I might fail."

| Facet | Status |
|---|---|
| Risktaking | `scored` |
| Perseverance | `insufficient_evidence` |
| Creative risk-taking tendency | `insufficient_evidence` |
| Comfort with Vulnerability | `insufficient_evidence` |
| Desperation | `insufficient_evidence` |

Conservative reasoning for each rejection:

- **Perseverance** — a single risky decision does not establish sustained effort or persistence through challenges over time.
- **Creative risk-taking tendency** — risk-taking is present, but creativity or innovative thinking is not directly established by this statement.
- **Comfort with Vulnerability** — the conversation does not directly establish comfort with vulnerability as a disposition.
- **Desperation** — the conversation does not establish urgent need, lack of alternatives, or a desperate state.

---

## 17. Decision: Retrieval Relevance vs Evidence Support

| Concept | Meaning | Decision Maker |
|---|---|---|
| Retrieval relevance | Semantic similarity between conversation and facet | Embedding model |
| Evidence support | Whether the conversation actually supports the facet | LLM |
| Final score | Deterministic mapping from evidence judgment | Python |

High retrieval similarity can still result in `insufficient_evidence`. This is **expected behavior**, not a system error — it is the direct consequence of separating candidate discovery from evidence-based judgment.

---

## 18. Decision: Why Not Score Directly from Similarity?

**Rejected alternative:**
Conversation → embeddings → similarity → score

Why this was rejected:

- Similarity is not behavioral evidence — it measures textual/semantic proximity, not truth.
- Related concepts can be confused (e.g., risk-taking vs. desperation) purely due to embedding-space closeness.
- High similarity can create false positives that look plausible but are not actually supported by the text.
- Numeric scores under this approach would depend on embedding geometry rather than explicit, inspectable evidence.

**Chosen approach:**
Conversation → retrieval → evidence evaluation → deterministic score

---

## 19. Decision: Why Not Send All Candidates in One LLM Prompt?

**Rejected alternative:**
Conversation + Facet 1 + Facet 2 + Facet 3 + … → one LLM call

Why this was deliberately not used:

- Risk of cross-facet contamination between judgments.
- Risk of anchoring, where the model's judgment on one facet influences its judgment on another.
- Risk of evidence leakage between facets sharing similar language.
- Less isolated attribution of evidence to a specific facet.
- Harder auditing, since a single combined output is less traceable per facet.

**Chosen approach:** one facet per LLM evaluation call.

The selected facets are not sent to the LLM together in one combined prompt:

```text
Conversation + Facet 1 → LLM → Result 1
Conversation + Facet 2 → LLM → Result 2
Conversation + Facet 3 → LLM → Result 3
```

This does not claim that batched evaluation is always the wrong choice in general — it states that independent evaluation was chosen specifically because this project's core requirement is strict, isolated evidence grounding per facet.

---

## 20. Decision: Why Use a Candidate Cap?

This section documents the explicit recall-vs-cost trade-off behind capping candidates at 5.

**Without a cap:**

- More candidates reach the LLM.
- More computation and cost per conversation.
- More marginal, low-confidence candidates enter evaluation.
- Harder to keep evaluation focused.

**With a cap:**

- Controlled, predictable evaluation workload.
- Focused set of the most promising candidates.
- Lower downstream computational cost.

**Accepted limitation:** some relevant facets may be excluded before ever reaching LLM evaluation. This is a known and accepted consequence of the cap, not a hidden flaw.

---

## 21. Decision: Why Deterministic Scoring Instead of LLM-Generated Scores?

**Rejected approach:** asking the LLM to directly output a numeric score.

**Chosen approach:**
LLM → qualitative evidence strength
Python → deterministic score/confidence

Benefits of the chosen approach:

- Consistent mapping from evidence strength to score across all runs.
- Auditable logic, since the mapping lives in inspectable Python code rather than free-form model output.
- Less dependence on the LLM's ability to generate well-calibrated numbers directly.
- Scoring policy can be changed independently of the evidence-evaluation prompt/model.

---

## 22. Failure Modes and Accepted Limitations

Only actual, known limitations are documented here.

**Retrieval limitation**
`all-MiniLM-L6-v2` may miss facets whose wording differs substantially from the conversation's phrasing.

**Threshold limitation**
The 0.25 similarity threshold can remove a relevant facet before it ever reaches LLM evaluation.

**Candidate-cap limitation**
Only five candidates reach the LLM evaluation stage per conversation.

**LLM limitation**
`Qwen/Qwen3-8B` reasoning consistency can affect evaluation results across different runs.

**Dataset limitation**
The system is limited to the supplied 399-facet library and cannot evaluate facets outside of it.

**Evaluation limitation**
The project has tested examples and manual validation, but does not currently have a large human-labeled benchmark with precision/recall/F1/calibration metrics.

No additional failure modes are claimed as implemented facts beyond those listed above.

---

## 23. Tested Evidence Behavior

### Example 1

**Conversation:**
> "I was terrified of failing, but I decided to start the company anyway."

**Result: Courageousness**

- status = `scored`
- score = 5
- confidence = 0.95
- evidence_strength = `very_strong`

This evidence supports **Courageousness** because the conversation contains both an explicit statement of fear and a deliberate action taken despite that fear — the defining pattern of courage.

Other semantically related facets retrieved for this conversation were **not** automatically scored; where evidence was insufficient, they were correctly marked `insufficient_evidence` rather than forced into a score.

### Example 2

**Conversation:**
> "I left my secure job and started a company even though I knew I might fail."

**Result: Risktaking**

- status = `scored`
- score = 5
- confidence = 0.95
- evidence_strength = `very_strong`

This evidence supports **Risktaking** because the speaker knowingly leaves security and accepts uncertainty and potential failure — direct evidence of accepting risk.

The following facets were marked `insufficient_evidence` for the same conversation:

- **Perseverance** — a single risky decision does not establish sustained effort or persistence through challenges.
- **Creative risk-taking tendency** — risk-taking is present, but creativity or innovative thinking is not directly established.
- **Comfort with Vulnerability** — the conversation does not directly establish comfort with vulnerability.
- **Desperation** — the conversation does not establish urgent need, lack of alternatives, or desperation.

These are **tested examples**, not benchmark performance metrics.

---

## 24. Repository and File Responsibilities
ai-facet-scoring/
├── data/
│ └── Facets Assignment.csv
├── src/
│ ├── pipeline.py
│ ├── preprocess.py
│ ├── retrieve.py
│ └── score.py
├── outputs/
├── .gitignore
├── README.md
└── decision.md

- **`data/Facets Assignment.csv`** — the 399-facet reference dataset.
- **`src/preprocess.py`** — conversation preprocessing (cleaning/normalization).
- **`src/retrieve.py`** — semantic retrieval, similarity threshold filtering, and action-aware candidate prioritization.
- **`src/score.py`** — independent per-facet LLM evaluation and deterministic score/confidence mapping.
- **`src/pipeline.py`** — end-to-end orchestration of preprocessing, retrieval, and scoring.
- **`outputs/`** — generated local artifacts (e.g. audit data, enriched facets, cached embeddings), produced when the pipeline runs. This directory is ignored by Git via `outputs/*` in `.gitignore`, so these generated files are **not committed to the repository**.
- **`.gitignore`** — excludes virtual environments, caches, generated outputs, and other ignored artifacts as configured in the project.
- **`README.md`** — user-facing project documentation.
- **`decision.md`** — this document; engineering rationale.

---

## 25. Engineering Trade-Off Summary

| Decision | Benefit | Trade-Off |
|---|---|---|
| Semantic retrieval | Efficient candidate discovery | Can retrieve semantically related false positives |
| Top 25 | Broad candidate recall | More candidates/noise before filtering |
| 0.25 threshold | Removes weak candidates | Can remove a relevant low-similarity facet |
| Action-aware prioritization | Favors evidence-verifiable candidates | Can prioritize concrete actions over abstract facets |
| Maximum 5 | Controls LLM workload | Some relevant facets may never reach LLM |
| Independent LLM calls | Evidence isolation | More LLM calls |
| Structured evidence output | Predictable downstream processing | Depends on model following output format |
| Deterministic scoring | Consistent and auditable scores | Requires a predefined mapping |
| Abstention | Prevents unsupported claims | Some plausible facets remain unscored |

---

## 26. Future Improvements

The following are identified **future work items** and are **NOT CURRENTLY IMPLEMENTED**:

1. Human-labeled automated evaluation benchmark — future work.
2. Retrieval precision/recall analysis — future work.
3. Threshold and top-k tuning — future work.
4. Improved candidate prioritization — future work.
5. Configurable top-k / threshold / candidate-cap parameters — future work.
6. Evaluation result caching — future work.
7. Better handling of longer and multi-topic conversations — future work.
8. Stronger structured-output validation — future work.
9. Systematic failure analysis across facet types — future work.
10. Comparison with additional retrieval models — future work.

None of these should be read as already implemented in the current project.

---

## 27. Final Engineering Rationale

The architecture of AI Facet Scoring reflects a consistent set of engineering commitments:

- Retrieve broadly enough to discover potentially relevant facets.
- Filter deliberately to control noise before expensive evaluation.
- Prioritize evidence-verifiable candidates over purely abstract ones.
- Evaluate every selected facet independently, in isolation from every other facet.
- Require direct conversational evidence before assigning any score.
- Keep numeric scoring deterministic and auditable in Python.
- Abstain explicitly when evidence is insufficient, rather than manufacturing a plausible-looking result.
- Accept known recall and cost trade-offs rather than pretending the system has perfect coverage.

**Retrieve broadly. Evaluate independently. Score deterministically. Reject unsupported claims.**

A facet is scored because the conversation provides evidence for it — not merely because the conversation sounds similar to it.