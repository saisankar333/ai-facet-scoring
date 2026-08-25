DEBUGGING.md — AI Facet Scoring

This document records actual problems encountered during development of the AI Facet Scoring pipeline, why each mattered, what was changed in response, and how each change was verified. It complements README.md (what the system does), DECISIONS.md (why the architecture is built this way), and PROMPT_LOG.md (evidence-evaluation prompt design).

1. Qwen Response Contained Separate Reasoning Content

During testing of Qwen/Qwen3-8B, responses were observed to contain two distinct parts:

message.content — the model's final answer.
message.reasoning_content — intermediate reasoning produced before the final answer.

reasoning_content must not be parsed as the final answer: it is free-form reasoning text, not guaranteed to be valid JSON, and is not the field the model is instructed to place its structured evidence judgment in. Treating it as the answer would risk parsing failures or picking up an unfinished or exploratory line of reasoning instead of the model's actual conclusion.

src/score.py was changed to read only message.content when extracting the structured evidence judgment. Debug output was subsequently reduced to show only the final model content, rather than printing the full reasoning trace, to keep debugging output focused on what is actually parsed.

2. Model Output Must Be Validated Before Scoring

The model is expected to return a structured JSON object with exactly four fields:

status
evidence_strength
evidence
reason

src/score.py performs JSON extraction and validation on message.content before accepting a result. This includes:

Direct JSON parsing of message.content.
Removal of Markdown code fences (e.g. ```json blocks) if the model wraps its output in them.
Extraction of a complete JSON object from the response text.
Validation that all required fields are present.
Validation that status is one of the allowed values (scored, insufficient_evidence, not_observable).
Validation that evidence_strength is one of the allowed values when present.
A requirement that evidence be present and non-empty when status is scored.

No additional validation beyond what is implemented in src/score.py is claimed here.

3. Missing Final Model Content

It was observed that Qwen/Qwen3-8B can, in some cases, produce reasoning content while the final message.content is missing or unusable (empty, or not parseable as the expected JSON structure).

In this situation:

reasoning_content is not used as a substitute for the missing final answer.
The pipeline does not fabricate a result to fill the gap.
Missing or unusable final content instead triggers retry behavior for that specific facet evaluation call, rather than proceeding with an invalid or guessed result.
4. Model Inference Failure

During testing, an inference failure was observed with the following error:

Server disconnected without sending a response.

This occurred while evaluating the facet Character strength: Perseverance. The call did not return any response content to parse.

src/score.py classifies this as a model_error rather than allowing it to be treated as a valid evaluation outcome. This keeps infrastructure/network failures clearly distinguished from legitimate evidence outcomes (scored, insufficient_evidence, not_observable) — a failed call is never silently converted into a score.

5. Hugging Face Authentication and Credit Errors

src/score.py includes error-handling paths for the following conditions when calling the Hugging Face inference endpoint for Qwen/Qwen3-8B:

Missing HF_TOKEN (no token configured in the environment).
Authentication failure (an invalid or rejected token).
Exhausted inference credits.
Unsupported model or provider configuration.

Each of these is handled as a distinct error condition rather than being allowed to crash the pipeline unhandled or produce a misleading result. No token value, real or example, is included in this document or in the repository.

6. Retrieval and Evidence Scoring Are Separate

The pipeline's architecture keeps candidate discovery and evidence evaluation strictly separate:

Semantic retrieval
     ↓
Candidate selection
     ↓
LLM evidence evaluation
     ↓
Deterministic scoring

This was directly confirmed using the tested conversation:

"I left my secure job and started a company even though I knew I might fail."

Retrieval surfaced several semantically related facets, but only Risktaking was actually scored. The following were observed as insufficient_evidence:

Perseverance
Creative risk-taking tendency
Comfort with Vulnerability
Desperation

This confirms that semantic similarity is not treated as evidence — a facet being retrieved as relevant does not, by itself, result in a score.

7. Perseverance Must Not Be Inferred From a Single Risky Decision

Using the same tested conversation:

"I left my secure job and started a company even though I knew I might fail."

Observed results:

Facet	Status
Risktaking	scored
Perseverance	insufficient_evidence

The conversation describes a single decision made at one point in time. It does not describe sustained effort through repeated setbacks, which is required to support Perseverance. This confirms the evaluation stage does not generalize a single risky action into a broader claim of persistence.

8. Perseverance Requires Sustained Evidence

Using the tested conversation:

"I kept working on my company for three years despite repeated failures and setbacks."

Observed results:

Facet	Status	Evidence strength	Score	Confidence
Perseverance	scored	very_strong	5	0.95
Character strength: Perseverance	scored	very_strong	5	0.95

This wording provides direct evidence of sustained effort through repeated setbacks: an explicit time span ("three years") combined with an explicit description of continuing despite failures. This is a materially different evidentiary pattern from the single-decision example in Section 7, and the evaluation stage correctly distinguished between them.

9. Perseverance Must Not Automatically Become Hardworking

Using the same three-year conversation, the following was observed:

Facet	Status
Perseverance	scored
Hardworking	insufficient_evidence

Persistence through setbacks over time does not automatically establish the broader Hardworking facet. The conversation provides direct evidence of continuing despite failure, but does not describe effort intensity, work habits, or diligence in a way that would directly support Hardworking as a separate facet. This confirms the evaluation stage treats semantically adjacent facets independently rather than assuming one implies the other.

10. Medical or Physical Facets Can Be Not Observable

Using the same three-year conversation, the following was observed:

Facet	Status
Burnout Symptoms	not_observable

The conversation contains no reference to exhaustion, stress symptoms, or wellbeing of any kind. Ordinary conversation describing effort and setbacks does not automatically establish medical or physical symptoms. This result is not, and is not intended as, any form of medical or psychological diagnosis — it simply reflects that the conversation provides no basis to evaluate that facet at all.

11. Insufficient Evidence vs Not Observable

Two distinct negative outcomes exist, and testing confirmed the pipeline distinguishes between them correctly:

insufficient_evidence — the facet is relevant to the conversation's general topic, but the specific evidence required to support it is not present. Example: Perseverance for the single-decision conversation (Section 7) — the topic (a company decision) is related, but sustained effort is not described.
not_observable — the conversation provides no basis to evaluate the facet at all. Example: Burnout Symptoms for the three-year conversation (Section 10) — nothing in the text relates to symptoms or wellbeing.

The distinction matters because it separates "this facet is plausible but unproven" from "this facet has nothing in the text to evaluate against."

12. Candidate Selection Debugging

The following command was used to inspect retrieval and candidate-selection behavior for a conversation without requiring a Hugging Face inference call:

bash
python src/pipeline.py --local-test

Observed configuration and results from this mode:

399 facets loaded.
TOP_K = 25.
Similarity threshold = 0.25.
Maximum LLM candidates = 5.
25 candidates retrieved.
A subset of those candidates passed the similarity threshold.
A final candidate-selection output listing the facets chosen for LLM evaluation.

This mode does not make a Hugging Face inference request — it stops after candidate selection, allowing retrieval and prioritization behavior to be inspected in isolation from LLM evidence evaluation.

13. Candidate Selection Uses Retrieval Plus Action Signals

src/pipeline.py uses semantic similarity together with a limited set of action/concept keyword signals when selecting and prioritizing candidates for LLM evaluation.

This is important to state precisely: these action-keyword matches are candidate-selection signals only. They influence which facets are prioritized to reach the LLM, but they are not final evidence and do not by themselves determine whether a facet is scored. Final evidence determination happens exclusively in src/score.py, based on the LLM's independent evaluation of the conversation against that single facet.

14. Deterministic Score and Confidence Mapping

src/score.py maps evidence_strength to a numeric score and confidence deterministically:

Evidence strength	Score	Confidence
very_weak	1	0.50
weak	2	0.65
moderate	3	0.75
strong	4	0.90
very_strong	5	0.95

For insufficient_evidence:

score      = null
confidence = 0.0

For not_observable:

score      = null
confidence = 1.0

Numeric scoring is performed entirely by Python, using this fixed mapping — it is not generated by the LLM. This keeps the score/confidence values consistent and reproducible for a given evidence-strength or status outcome, independent of any variance in the model's free-form generation.

15. Verification Performed During Debugging

The following commands were actually used during development to verify behavior:

bash
python -m py_compile src/score.py
bash
python src/pipeline.py --local-test
bash
python src/pipeline.py "I left my secure job and started a company even though I knew I might fail."
bash
python src/pipeline.py "I kept working on my company for three years despite repeated failures and setbacks."

In addition, debug output handling was changed so that only the final model content (message.content) is printed during runs, rather than the full response including reasoning_content, to keep debugging output aligned with what is actually parsed and scored.

16. Current Debugging Principle

The debugging history above reflects one consistent underlying principle:

Retrieval finds candidates. The LLM evaluates direct evidence. Python determines the numeric score.

At every stage where this separation could have been blurred — parsing reasoning as an answer, treating candidate-selection keyword matches as evidence, generalizing one scored facet into a related but unproven one, or letting a failed inference call pass as a valid result — the implementation was checked and, where necessary, corrected to preserve the boundary between these responsibilities.

The system distinguishes four categories of outcome for every facet evaluation:

Valid evidence (scored) — direct support found, deterministically mapped to a score and confidence.
Insufficient evidence (insufficient_evidence) — relevant but unsupported.
Not observable (not_observable) — no basis to evaluate at all.
Model/inference errors (model_error) — a failure in the call itself, never treated as a scoring outcome.

This document does not claim the system is free of limitations; known limitations and accepted trade-offs are documented separately in DECISIONS.md.
