import json
import os
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import InferenceClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACET_PATH = PROJECT_ROOT / "outputs" / "enriched_facets.csv"

MODEL_NAME = "Qwen/Qwen3-8B"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a conservative conversation facet evaluator.

Your job is to determine whether the provided conversation contains
DIRECT evidence for the specified facet.

You must evaluate ONLY the facet provided by the user.

============================================================
CORE PRINCIPLE
============================================================

A facet must be supported by the conversation itself.

Do NOT treat semantic similarity between the facet name and the
conversation as evidence.

Do NOT infer a personality trait merely because an action could be
associated with that trait.

One observed action must NOT automatically be converted into
multiple personality traits.

============================================================
EVIDENCE RULES
============================================================

1. Never invent evidence.

2. Never use outside knowledge.

3. Evidence must come directly from the conversation.

4. Only return "scored" when the conversation directly supports
   the facet.

5. If the evidence is only an indirect interpretation, return
   "insufficient_evidence".

6. If the conversation contains no meaningful direct evidence
   for the facet, return "insufficient_evidence".

7. Do not infer motives that were not explicitly stated.

8. Do not infer emotions that were not explicitly stated.

9. Do not infer relationships that were not explicitly stated.

10. Do not infer outcomes that were not explicitly stated.

11. Do not infer repeated behavior from a single event.

12. Do not infer a stable personality trait from one action unless
    the action directly expresses the facet.

13. If conversation_observable=false, return "not_observable".

14. If conversation_observable=true but direct evidence is
    insufficient, return "insufficient_evidence".

15. Evidence must quote or closely summarize information actually
    present in the conversation.

============================================================
IMPORTANT EXAMPLES
============================================================

Conversation:

"I left my secure job and started a company even though I knew
I might fail."

This directly supports:

- Risktaking

It does NOT by itself establish:

- Perseverance
- Resilience
- Desperation
- Trust in others
- Courageousness
- Decision-making confidence
- Comfort with vulnerability
- Leadership
- Creativity

Those facets require additional evidence unless the conversation
explicitly supports them.

------------------------------------------------------------

Conversation:

"I kept working on my company for three years despite repeated
failures and setbacks."

This directly supports:

- Perseverance

The continuation of effort over three years despite repeated
failures is direct evidence of perseverance.

------------------------------------------------------------

Conversation:

"I relied on my co-founder and trusted her judgment."

This directly supports:

- Trust in others

------------------------------------------------------------

Conversation:

"I was terrified but decided to do it anyway."

This provides direct evidence related to:

- Courageousness

because fear and the decision to act despite that fear are explicitly
stated.

============================================================
IMPORTANT DISTINCTIONS
============================================================

Risktaking:

A directly described action involving knowingly accepting uncertainty
or potential loss can support Risktaking.

Perseverance:

Requires continuing effort, persistence, or sustained action despite
difficulty, failure, setbacks, or obstacles.

Courageousness:

Do not infer courage merely from a risky action.

Explicit fear, bravery, overcoming fear, or explicitly described
courage-related behavior provides stronger direct evidence.

Trust in others:

Requires explicit evidence of trusting, relying on, depending on,
or placing confidence in another person.

Decision-making confidence:

Requires direct evidence concerning confidence, certainty, trust,
or belief in one's own decision-making.

Do not infer decision-making confidence merely because someone made
a difficult or risky decision.

Hardworking:

Do not automatically infer Hardworking from Perseverance.

"Kept working despite failures" is strong evidence for Perseverance.
It may not independently establish the broader trait Hardworking
unless the conversation directly describes hard work, diligence,
substantial effort, or similar behavior.

============================================================
EVIDENCE STRENGTH
============================================================

If the facet is directly supported, classify the evidence as exactly
one of:

"very_weak"
"weak"
"moderate"
"strong"
"very_strong"

Definitions:

very_weak:
Barely direct evidence, but still enough to score.

weak:
Limited direct evidence.

moderate:
Clear direct evidence, but limited in scope.

strong:
Clear and substantial direct evidence.

very_strong:
Explicit, unambiguous, highly specific direct evidence.

Do NOT use evidence_strength when evidence is insufficient.

============================================================
OUTPUT FORMAT
============================================================

Return valid JSON ONLY.

Do NOT return Markdown.

Do NOT return code fences.

Do NOT return analysis.

Do NOT return reasoning.

Do NOT return score.

Do NOT return confidence.

Python will calculate score and confidence from evidence_strength.

------------------------------------------------------------
For scored:

{
  "status": "scored",
  "evidence_strength": "very_strong",
  "evidence": "short direct evidence from conversation",
  "reason": "short explanation"
}

------------------------------------------------------------
For insufficient evidence:

{
  "status": "insufficient_evidence",
  "evidence_strength": null,
  "evidence": "",
  "reason": "short explanation of what direct evidence is missing"
}

------------------------------------------------------------
For not observable:

{
  "status": "not_observable",
  "evidence_strength": null,
  "evidence": "",
  "reason": "short explanation"
}

Return exactly one JSON object.
"""


# ============================================================
# DETERMINISTIC SCORE MAPPING
# ============================================================

EVIDENCE_STRENGTH_MAP = {
    "very_weak": {
        "score": 1,
        "confidence": 0.50,
    },
    "weak": {
        "score": 2,
        "confidence": 0.65,
    },
    "moderate": {
        "score": 3,
        "confidence": 0.75,
    },
    "strong": {
        "score": 4,
        "confidence": 0.90,
    },
    "very_strong": {
        "score": 5,
        "confidence": 0.95,
    },
}


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(conversation, facet):

    return f"""
Conversation:
{conversation}

Facet:
{facet["raw_facet"]}

Facet type:
{facet["facet_type"]}

Conversation observable:
{facet["conversation_observable"]}

Evaluate ONLY the facet above.

Use only direct evidence from the conversation.

Do not infer unsupported traits.

Do not infer motives.

Do not infer emotions.

Do not infer relationships.

Do not infer outcomes.

Do not infer repeated behavior unless explicitly stated.

If direct evidence is insufficient, return:

{{
  "status": "insufficient_evidence",
  "evidence_strength": null,
  "evidence": "",
  "reason": "short explanation"
}}

If directly supported, return:

{{
  "status": "scored",
  "evidence_strength": "very_strong",
  "evidence": "direct evidence from the conversation",
  "reason": "short explanation"
}}

Choose evidence_strength carefully.

Return ONLY valid JSON.
"""


# ============================================================
# MODEL ERROR
# ============================================================

def model_error(reason):

    return {
        "status": "model_error",
        "score": None,
        "confidence": 0.0,
        "evidence": "",
        "reason": reason,
    }


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_object(text):

    if text is None:
        return None

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if not text:
        return None

    # --------------------------------------------------------
    # DIRECT JSON
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # REMOVE MARKDOWN CODE FENCES
    # --------------------------------------------------------

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.replace(
        "```",
        "",
    ).strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # FIND FIRST COMPLETE JSON OBJECT
    # --------------------------------------------------------

    start = cleaned.find("{")

    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(cleaned)):

        char = cleaned[i]

        if escape:

            escape = False
            continue

        if char == "\\":

            escape = True
            continue

        if char == '"':

            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":

            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                candidate = cleaned[
                    start:i + 1
                ]

                try:

                    return json.loads(
                        candidate
                    )

                except json.JSONDecodeError:

                    return None

    return None


# ============================================================
# VALIDATE + CONVERT MODEL RESULT
# ============================================================

def parse_result(text):

    if text is None:

        return model_error(
            "Model returned no final text content."
        )

    data = extract_json_object(
        text
    )

    if data is None:

        return model_error(
            "Model returned invalid or truncated JSON."
        )

    if not isinstance(data, dict):

        return model_error(
            "Model output must be a JSON object."
        )

    # --------------------------------------------------------
    # REQUIRED MODEL FIELDS
    # --------------------------------------------------------

    required = {
        "status",
        "evidence_strength",
        "evidence",
        "reason",
    }

    missing = required - set(
        data.keys()
    )

    if missing:

        return model_error(
            f"Missing required fields: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    allowed_statuses = {
        "scored",
        "insufficient_evidence",
        "not_observable",
    }

    status = data["status"]

    if status not in allowed_statuses:

        return model_error(
            f"Invalid status: {status}"
        )

    # --------------------------------------------------------
    # TEXT FIELDS
    # --------------------------------------------------------

    evidence = str(
        data["evidence"]
    ).strip()

    reason = str(
        data["reason"]
    ).strip()

    # ========================================================
    # SCORED
    # ========================================================

    if status == "scored":

        evidence_strength = data[
            "evidence_strength"
        ]

        if evidence_strength not in EVIDENCE_STRENGTH_MAP:

            return model_error(
                "Invalid evidence_strength: "
                f"{evidence_strength}"
            )

        # Scored results must contain evidence.

        if not evidence:

            return model_error(
                "Scored result must contain direct evidence."
            )

        mapping = EVIDENCE_STRENGTH_MAP[
            evidence_strength
        ]

        score = mapping[
            "score"
        ]

        confidence = mapping[
            "confidence"
        ]

        return {
            "status": "scored",
            "score": score,
            "confidence": confidence,
            "evidence": evidence,
            "reason": reason,
        }

    # ========================================================
    # INSUFFICIENT EVIDENCE
    # ========================================================

    if status == "insufficient_evidence":

        return {
            "status": "insufficient_evidence",
            "score": None,
            "confidence": 0.0,
            "evidence": "",
            "reason": reason,
        }

    # ========================================================
    # NOT OBSERVABLE
    # ========================================================

    return {
        "status": "not_observable",
        "score": None,
        "confidence": 1.0,
        "evidence": "",
        "reason": reason,
    }


# ============================================================
# DEBUG RESPONSE
# ============================================================

def debug_message(message):

    # Keep debug output limited to the final answer content.
    # Do not print model reasoning or the complete response object.
    content = getattr(
        message,
        "content",
        None,
    )

    print("\nMODEL FINAL CONTENT:")
    print(repr(content))


# ============================================================
# MODEL ERROR CLASSIFICATION
# ============================================================

def classify_inference_error(exc):

    error_text = str(exc)

    # --------------------------------------------------------
    # HUGGING FACE CREDIT ERROR
    # --------------------------------------------------------

    if (
        "402" in error_text
        or "Payment Required" in error_text
        or "depleted your monthly included credits"
        in error_text
    ):

        return (
            "Hugging Face inference credits are exhausted. "
            "No facet score was produced."
        )

    # --------------------------------------------------------
    # MODEL PROVIDER ERROR
    # --------------------------------------------------------

    if (
        "model_not_supported" in error_text
        or "not supported by any provider"
        in error_text
    ):

        return (
            f"Model '{MODEL_NAME}' is not currently "
            "supported by the configured Hugging Face "
            "Inference Provider."
        )

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    if (
        "401" in error_text
        or "Unauthorized" in error_text
        or "Invalid token" in error_text
    ):

        return (
            "Hugging Face authentication failed. "
            "Check HF_TOKEN."
        )

    # --------------------------------------------------------
    # OTHER
    # --------------------------------------------------------

    return (
        f"Inference request failed: {error_text}"
    )


# ============================================================
# CALL MODEL
# ============================================================

def call_model(
    client,
    conversation,
    facet,
):

    try:

        response = client.chat_completion(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_prompt(
                        conversation,
                        facet,
                    ),
                },
            ],

            temperature=0.0,

            max_tokens=800,

            response_format={
                "type": "json_object",
            },
        )

    except Exception as exc:

        return model_error(
            classify_inference_error(
                exc
            )
        )

    # --------------------------------------------------------
    # EXTRACT MESSAGE
    # --------------------------------------------------------

    try:

        message = response.choices[
            0
        ].message

    except Exception as exc:

        return model_error(
            f"Could not read model response: {exc}"
        )

    debug_message(
        message
    )

    # --------------------------------------------------------
    # FINAL CONTENT ONLY
    # --------------------------------------------------------

    content = getattr(
        message,
        "content",
        None,
    )

    # IMPORTANT:
    #
    # reasoning_content is NOT treated as the answer.
    #
    # Qwen3 may spend tokens generating reasoning but return
    # content=None. We retry instead of parsing reasoning.
    #

    if content is None:

        return None

    content = str(
        content
    ).strip()

    if not content:

        return None

    return parse_result(
        content
    )


# ============================================================
# RETRY MODEL
# ============================================================

def retry_model(
    client,
    conversation,
    facet,
):

    print(
        "\nModel did not return usable final JSON."
    )

    print(
        "Retrying with explicit final-JSON instruction..."
    )

    retry_prompt = f"""
Conversation:
{conversation}

Facet:
{facet["raw_facet"]}

Facet type:
{facet["facet_type"]}

Conversation observable:
{facet["conversation_observable"]}

Return ONLY one valid JSON object.

DO NOT output reasoning.

DO NOT output analysis.

DO NOT output Markdown.

DO NOT output code fences.

DO NOT output score.

DO NOT output confidence.

Use this exact schema:

{{
  "status": "scored",
  "evidence_strength": "very_strong",
  "evidence": "direct evidence from the conversation",
  "reason": "short explanation"
}}

Allowed evidence_strength values:

"very_weak"
"weak"
"moderate"
"strong"
"very_strong"

If there is insufficient direct evidence, return:

{{
  "status": "insufficient_evidence",
  "evidence_strength": null,
  "evidence": "",
  "reason": "short explanation"
}}

If conversation_observable is false, return:

{{
  "status": "not_observable",
  "evidence_strength": null,
  "evidence": "",
  "reason": "short explanation"
}}

IMPORTANT:

Only use evidence explicitly present in the conversation.

Do not infer personality traits from unrelated actions.

Return JSON ONLY.
"""

    try:

        response = client.chat_completion(

            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": retry_prompt,
                },
            ],

            temperature=0.0,

            max_tokens=800,

            response_format={
                "type": "json_object",
            },
        )

    except Exception as exc:

        return model_error(
            "Retry inference failed: "
            f"{classify_inference_error(exc)}"
        )

    try:

        message = response.choices[
            0
        ].message

    except Exception as exc:

        return model_error(
            f"Could not read retry response: {exc}"
        )

    debug_message(
        message
    )

    content = getattr(
        message,
        "content",
        None,
    )

    if content is None:

        return None

    content = str(
        content
    ).strip()

    if not content:

        return None

    return parse_result(
        content
    )


# ============================================================
# SCORE FACET
# ============================================================

def score_facet(
    client,
    conversation,
    facet,
):

    # ========================================================
    # DETERMINISTIC OBSERVABILITY GATE
    # ========================================================

    observable = facet[
        "conversation_observable"
    ]

    # Handle CSV boolean values safely.

    if isinstance(observable, str):

        observable = (
            observable.strip().lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

    else:

        observable = bool(
            observable
        )

    if not observable:

        return {
            "status": "not_observable",
            "score": None,
            "confidence": 1.0,
            "evidence": "",
            "reason": str(
                facet.get(
                    "abstention_reason",
                    "Facet is not observable from conversation.",
                )
            ),
        }

    # ========================================================
    # FIRST MODEL CALL
    # ========================================================

    result = call_model(
        client,
        conversation,
        facet,
    )

    if result is not None:

        return result

    # ========================================================
    # RETRY
    # ========================================================

    result = retry_model(
        client,
        conversation,
        facet,
    )

    if result is not None:

        return result

    # ========================================================
    # FINAL FAILURE
    # ========================================================

    return model_error(
        "Model returned no valid final JSON after retry."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # HUGGING FACE TOKEN
    # ========================================================

    token = os.getenv(
        "HF_TOKEN"
    )

    if not token:

        raise RuntimeError(
            "HF_TOKEN environment variable is not set."
        )

    # ========================================================
    # FACET FILE
    # ========================================================

    if not FACET_PATH.exists():

        raise FileNotFoundError(
            f"Facet file not found: {FACET_PATH}"
        )

    facets = pd.read_csv(
        FACET_PATH
    )

    # ========================================================
    # CLIENT
    # ========================================================

    client = InferenceClient(
        api_key=token
    )

    # ========================================================
    # TEST CONVERSATION
    # ========================================================

    conversation = (
        "I left my secure job and started a company "
        "even though I knew I might fail."
    )

    # ========================================================
    # TEST FACETS
    # ========================================================

    test_facets = facets[
        facets[
            "normalized_facet"
        ].isin(
            [
                "risktaking",
                "perseverance",
                "trust in others",
                "courageousness",
                "decision-making confidence",
            ]
        )
    ]

    print("\nModel:")
    print(
        MODEL_NAME
    )

    print("\nConversation:")
    print(
        conversation
    )

    print(
        f"\nNumber of test facets: "
        f"{len(test_facets)}"
    )

    print("\nResults:")

    # ========================================================
    # SCORE EACH FACET
    # ========================================================

    for _, facet in test_facets.iterrows():

        print(
            f"\nScoring facet: "
            f"{facet['raw_facet']}"
        )

        result = score_facet(
            client,
            conversation,
            facet,
        )

        output = {
            "facet": facet[
                "raw_facet"
            ],
            **result,
        }

        print(
            json.dumps(
                output,
                indent=2,
                ensure_ascii=False,
            )
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()