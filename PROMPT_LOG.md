PROMPT_LOG.md — Evidence-Evaluation Prompt Development

This document records the design, rules, and observed behavior of the LLM evidence-evaluation prompt used in the AI Facet Scoring pipeline. It complements decision.md (architectural rationale) and the README.md (system overview) by focusing specifically on how the per-facet evaluation prompt was designed, what it forbids, what it returns, and how its output was actually observed to behave during testing.

Repository: https://github.com/saisankar333/ai-facet-scoring

1. Core System-Prompt Objective

The evidence-evaluation prompt sent to Qwen/Qwen3-8B has a single, narrow objective per call:

Given a conversation and one facet definition, determine whether the conversation contains direct evidence that supports that facet — and if so, how strong that evidence is.

The prompt is not asked to describe the person, summarize the conversation, or produce a personality profile. It is asked one narrow, evidence-focused question at a time. This objective is the basis for every other rule in the prompt.

2. Rule: Semantic Relevance Is Not Evidence

The prompt explicitly instructs the model that a facet being retrieved (i.e., semantically similar to the conversation) does not mean the facet is true.

The model is told to ignore how "related" or "on-topic" a facet feels and instead ask only:

"Does the conversation contain a direct statement or action that supports this specific facet?"

This rule exists because retrieval (via all-MiniLM-L6-v2) already guarantees that every candidate reaching the LLM is topically related. If the LLM also scored based on relatedness, the evidence-evaluation stage would add no value beyond retrieval. The prompt is written to force a second, independent judgment rather than rubber-stamping retrieval's output.

3. Rule: Evaluate Only One Facet at a Time

The prompt is invoked once per candidate facet. It never contains more than one facet definition in a given call.

Conversation + Facet 1 → LLM → Result 1
Conversation + Facet 2 → LLM → Result 2
Conversation + Facet 3 → LLM → Result 3

This is a hard constraint of the prompt design, not just a pipeline convenience: the prompt is written assuming a single-facet context, and its instructions ("this facet," "the facet above") depend on there being exactly one facet under evaluation. This avoids cross-facet contamination and keeps each judgment auditable in isolation (see decision.md, Section 11 and Section 19).

4. Rule: Use Only Direct Evidence From the Conversation

The prompt requires that any evidence cited must be traceable to actual text in the conversation. The model is instructed to quote or closely paraphrase the specific portion of the conversation that supports the facet, rather than describing the facet in the abstract.

If no such directly traceable evidence exists, the correct response is not a low score — it is insufficient_evidence (see Section 7).

5. Prohibition Against Unsupported Inference

The prompt explicitly prohibits the model from inferring the following unless they are explicitly stated in the conversation:

Motives — why the speaker did something, if not stated.
Emotions — feelings not directly expressed.
Relationships — connections to other people not described.
Outcomes — results or consequences not mentioned.
Repeated behavior — patterns or habits, unless the conversation explicitly describes repetition or duration.
Unsupported personality traits — broader dispositions not directly evidenced by the specific statement.

This rule is the direct enforcement mechanism behind the project's core principle: semantic similarity finds candidates, evidence determines the score. Without this prohibition, the model would tend to "fill in" a plausible backstory around a short conversation snippet, which would reintroduce exactly the kind of unsupported behavioral claim the system is designed to avoid.

6. Structured Output Fields

Every evaluation call returns exactly four fields:

json
{
  "status": "scored",
  "evidence_strength": "very_strong",
  "evidence": "...",
  "reason": "..."
}
Field	Purpose
status	Whether the facet was scored, lacked sufficient evidence, or was not observable at all
evidence_strength	A qualitative rating of how strong the cited evidence is
evidence	The specific conversational content supporting the facet (empty when not applicable)
reason	A short explanation of the model's judgment

No other fields are requested from the LLM.

7. Allowed Status Values
Status	Meaning
scored	Direct evidence exists in the conversation supporting the facet
insufficient_evidence	The facet is plausible/relevant, but the conversation does not directly support it
not_observable	The conversation provides no basis whatsoever to evaluate the facet

insufficient_evidence and not_observable are distinct outcomes: the former acknowledges relevance without support, while the latter indicates the conversation simply doesn't touch on anything related to the facet.

8. Allowed Evidence Strength Values

When status is scored, the model selects one of five ordered evidence-strength levels:

very_weak
weak
moderate
strong
very_strong

These values are the only qualitative signal the LLM provides about the strength of a match. They are the sole input to the deterministic scoring step described below.

9. Score and Confidence Are Intentionally Not Requested From the LLM

The prompt deliberately does not ask the model to produce a numeric score or a confidence value.

This is a deliberate design choice, not an oversight:

Free-form numeric generation from an LLM is inconsistent across runs and hard to audit.
Asking for a qualitative judgment (evidence_strength) and mapping it deterministically in code keeps the scoring policy separate from the model's reasoning.
It allows the scoring logic to be reviewed, tested, and changed without touching the prompt or re-querying the model.

The LLM's job ends at producing a structured evidence judgment. Everything numeric happens downstream in Python.

10. Deterministic Python Mapping: Evidence Strength → Score/Confidence

Once a scored result is returned, src/score.py deterministically maps the evidence_strength value to a final numeric score and confidence value. The same evidence_strength value always produces the same score/confidence pair, since the mapping is fixed code rather than a per-call model decision.

For insufficient_evidence and not_observable results, no mapping is applied:

status     = "insufficient_evidence"
score      = null
confidence = 0.0

No additional scoring formula beyond this deterministic mapping is documented here, consistent with decision.md (Section 15), since no other formula has been implemented.

11. Tested Example — Risktaking

Conversation:

"I left my secure job and started a company even though I knew I might fail."

Observed evaluation results:

Facet	Status
Risktaking	scored
Perseverance	insufficient_evidence
Creative risk-taking tendency	insufficient_evidence
Comfort with Vulnerability	insufficient_evidence
Desperation	insufficient_evidence

Risktaking was scored because the statement directly and explicitly describes leaving security and knowingly accepting the possibility of failure. The remaining candidates were correctly marked insufficient_evidence: the conversation contains a single decision, not evidence of sustained effort (Perseverance), creative thinking (Creative risk-taking tendency), comfort with exposure over time (Comfort with Vulnerability), or urgency/lack of alternatives (Desperation).

12. Tested Example — Perseverance

Conversation:

"I kept working on my company for three years despite repeated failures and setbacks."

Observed evaluation results:

Facet	Status
Perseverance	scored
Character strength: Perseverance	scored
Hardworking	insufficient_evidence
Inefficiency	insufficient_evidence
Burnout Symptoms	not_observable

Both Perseverance and Character strength: Perseverance were scored because the statement explicitly describes sustained effort over a defined period ("three years") in the face of repeated setbacks — direct evidence of persistence through difficulty, unlike the single-decision example above.

Hardworking and Inefficiency were marked insufficient_evidence: the conversation describes persistence, not effort intensity or effectiveness/inefficiency specifically. Burnout Symptoms was marked not_observable, since the conversation contains no reference to exhaustion, stress symptoms, or wellbeing at all — there is nothing in the text to evaluate that facet against, as opposed to a facet that is topically close but unsupported.

13. What These Two Examples Demonstrate

Together, the Risktaking and Perseverance examples illustrate the distinction the system is built around:

Semantic retrieval surfaces facets that are topically related to a conversation — in both examples, several facets around risk, effort, and resilience were retrieved as candidates.
Evidence-supported scoring only assigns a score to the facets that the specific wording of the conversation actually substantiates.

The same general theme (leaving a secure path, persisting through a company) produces different scored facets depending on what is explicitly stated. A single risky decision earns Risktaking, not Perseverance. Three years of sustained effort through setbacks earns Perseverance, not merely "Hardworking" or an inference about burnout that was never mentioned. This is direct, observed confirmation that the pipeline's evidence-evaluation stage is doing more than re-stating retrieval's relevance ranking.

14. Qwen3 Response-Handling: content vs reasoning_content

During testing, it was observed that Qwen/Qwen3-8B responses can include two distinct parts:

message.content — the model's final answer.
message.reasoning_content — intermediate reasoning the model produces before arriving at its answer.

The pipeline parses only message.content for the structured JSON evidence judgment. reasoning_content is not parsed as output.

This is necessary because:

The structured JSON schema (status, evidence_strength, evidence, reason) is expected in content, not in the reasoning trace.
Reasoning text is free-form and not guaranteed to be valid JSON, so attempting to parse it directly would be unreliable.
Keeping parsing scoped to content keeps the JSON-extraction logic simple and matches the field the model is instructed to place its final answer in.
15. Retry Behavior When Final JSON Content Is Missing

If a call to Qwen/Qwen3-8B does not return usable structured JSON in message.content (for example, if content is empty or does not parse as valid JSON), the pipeline retries the evaluation call for that facet rather than silently failing or fabricating a result.

This retry behavior exists specifically to handle the content/reasoning_content split described above — if a response's final answer is missing or malformed, retrying gives the model another opportunity to produce a well-formed structured response for that single facet, without affecting the independent evaluation of any other facet.

Summary

The evidence-evaluation prompt is built around one narrow question per call — "does the conversation directly support this one facet?" — enforced through explicit inference prohibitions, a fixed structured-output schema, and a strict separation between qualitative evidence judgment (LLM) and quantitative scoring (Python). The tested Risktaking and Perseverance examples demonstrate this in practice: facets are scored because specific conversational evidence supports them, not because they are topically related to the conversation.
