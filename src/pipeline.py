import json
import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from retrieve import (
    load_or_build_index,
    retrieve_facets,
    MODEL_NAME as RETRIEVAL_MODEL_NAME,
)

from score import (
    score_facet,
    MODEL_NAME as SCORING_MODEL_NAME,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACET_PATH = PROJECT_ROOT / "outputs" / "enriched_facets.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 25

SIMILARITY_THRESHOLD = 0.25

MAX_LLM_CANDIDATES = 5


# ============================================================
# LOAD FACETS
# ============================================================

def load_facets():

    if not FACET_PATH.exists():

        raise FileNotFoundError(
            f"Facet file not found: {FACET_PATH}"
        )

    facets = pd.read_csv(
        FACET_PATH
    )

    print(
        f"Loaded {len(facets)} facets."
    )

    return facets


# ============================================================
# LOAD RETRIEVAL MODEL
# ============================================================

def load_retrieval_model():

    print(
        f"Loading retrieval model: "
        f"{RETRIEVAL_MODEL_NAME}"
    )

    return SentenceTransformer(
        RETRIEVAL_MODEL_NAME
    )


# ============================================================
# RETRIEVE FACETS
# ============================================================

def retrieve_candidates(
    conversation,
    retrieval_model,
    facets,
):

    facet_embeddings = load_or_build_index(
        retrieval_model,
        facets
    )

    retrieved = retrieve_facets(
        conversation,
        retrieval_model,
        facets,
        facet_embeddings
    )

    print(
        f"Retrieved {len(retrieved)} candidates."
    )

    return retrieved


# ============================================================
# SIMILARITY THRESHOLD
# ============================================================

def apply_similarity_threshold(
    retrieved
):

    thresholded = retrieved[
        retrieved["similarity"] >= SIMILARITY_THRESHOLD
    ].copy()

    print(
        f"Candidates above similarity threshold "
        f"({SIMILARITY_THRESHOLD}): "
        f"{len(thresholded)}"
    )

    return thresholded


# ============================================================
# CANDIDATE SELECTION
# ============================================================

def select_llm_candidates(
    thresholded,
    conversation
):

    if thresholded.empty:

        return thresholded.copy()

    candidates = thresholded.copy()

    conversation_lower = (
        conversation.lower()
    )


    # ========================================================
    # DIRECT FACET NAME MATCH
    # ========================================================

    candidates["direct_name_match"] = (
        candidates["normalized_facet"]
        .fillna("")
        .str.lower()
        .apply(
            lambda facet_name:
                bool(facet_name)
                and facet_name in conversation_lower
        )
    )


    # ========================================================
    # ACTION / CONCEPT MATCHES
    #
    # IMPORTANT:
    #
    # These are ONLY candidate-selection signals.
    # They are NOT evidence.
    #
    # Final evidence determination happens inside score.py.
    # ========================================================

    action_keywords = {

        "risktaking": [
            "risk",
            "risky",
            "secure job",
            "might fail",
            "could fail",
            "uncertain",
            "uncertainty",
            "took a chance",
            "took the chance",
        ],

        "perseverance": [
            "kept going",
            "continued",
            "persisted",
            "despite setbacks",
            "despite failure",
            "despite failures",
            "kept working",
            "didn't give up",
            "did not give up",
            "for three years",
            "for years",
        ],

        "character strength: perseverance": [
            "kept going",
            "continued",
            "persisted",
            "despite setbacks",
            "despite failure",
            "despite failures",
            "kept working",
            "didn't give up",
            "did not give up",
        ],

        "persistence": [
            "kept going",
            "continued",
            "persisted",
            "despite setbacks",
            "despite failure",
            "despite failures",
            "kept working",
            "didn't give up",
            "did not give up",
        ],

        "courageousness": [
            "afraid",
            "scared",
            "terrified",
            "fear",
            "despite fear",
            "courage",
        ],

        "trust in others": [
            "trusted",
            "trust",
            "relied on",
            "depended on",
            "co-founder",
            "team",
        ],

        "decision-making confidence": [
            "confident",
            "confidence",
            "sure",
            "certain",
            "right decision",
            "confident decision",
        ],
    }


    def has_action_match(
        facet_name
    ):

        keywords = action_keywords.get(
            facet_name,
            []
        )

        return any(
            keyword in conversation_lower
            for keyword in keywords
        )


    candidates["action_match"] = (
        candidates["normalized_facet"]
        .fillna("")
        .str.lower()
        .apply(
            has_action_match
        )
    )


    # ========================================================
    # SELECTION PRIORITY
    # ========================================================

    candidates["selection_priority"] = (
        candidates["similarity"]
        + candidates[
            "direct_name_match"
        ].astype(float) * 1.0
        + candidates[
            "action_match"
        ].astype(float) * 0.5
    )


    # ========================================================
    # SORT
    # ========================================================

    candidates = candidates.sort_values(
        by=[
            "direct_name_match",
            "action_match",
            "similarity",
        ],
        ascending=[
            False,
            False,
            False,
        ]
    )


    # ========================================================
    # RESERVE ACTION-MATCH SLOTS
    # ========================================================

    action_matches = candidates[
        candidates["action_match"]
    ]

    normal_candidates = candidates[
        ~candidates["action_match"]
    ]


    reserved_action = (
        action_matches.head(2)
    )

    remaining_slots = (
        MAX_LLM_CANDIDATES
        - len(reserved_action)
    )


    if remaining_slots > 0:

        remaining = (
            normal_candidates.head(
                remaining_slots
            )
        )

        candidates = pd.concat(
            [
                reserved_action,
                remaining,
            ]
        )

    else:

        candidates = reserved_action


    # ========================================================
    # FINAL ORDER
    # ========================================================

    candidates = candidates.sort_values(
        by=[
            "action_match",
            "similarity",
        ],
        ascending=[
            False,
            False,
        ]
    )


    candidates = candidates.head(
        MAX_LLM_CANDIDATES
    ).copy()


    print(
        f"Selected {len(candidates)} "
        f"candidates for scoring."
    )

    return candidates


# ============================================================
# PREPARE CANDIDATES
# ============================================================

def prepare_candidates(
    conversation
):

    facets = load_facets()

    retrieval_model = (
        load_retrieval_model()
    )

    retrieved = retrieve_candidates(
        conversation,
        retrieval_model,
        facets
    )

    thresholded = (
        apply_similarity_threshold(
            retrieved
        )
    )

    candidates = (
        select_llm_candidates(
            thresholded,
            conversation
        )
    )

    return (
        facets,
        retrieved,
        thresholded,
        candidates
    )


# ============================================================
# BUILD RESULT
# ============================================================

def build_result(
    facet,
    result
):

    return {
        "facet": facet["raw_facet"],

        "normalized_facet":
            facet["normalized_facet"],

        "facet_type":
            facet["facet_type"],

        "conversation_observable":
            bool(
                facet[
                    "conversation_observable"
                ]
            ),

        "retrieval_similarity":
            float(
                facet["similarity"]
            ),

        **result,
    }


# ============================================================
# NORMAL PIPELINE
# ============================================================

def run_pipeline(
    conversation
):

    (
        facets,
        retrieved,
        thresholded,
        candidates
    ) = prepare_candidates(
        conversation
    )


    # ========================================================
    # NO CANDIDATES
    # ========================================================

    if candidates.empty:

        return {
            "conversation":
                conversation,

            "retrieval_model":
                RETRIEVAL_MODEL_NAME,

            "scoring_model":
                SCORING_MODEL_NAME,

            "top_k":
                TOP_K,

            "similarity_threshold":
                SIMILARITY_THRESHOLD,

            "retrieved_count":
                len(retrieved),

            "threshold_pass_count":
                len(thresholded),

            "llm_candidate_limit":
                MAX_LLM_CANDIDATES,

            "candidate_count":
                0,

            "results":
                [],
        }


    # ========================================================
    # HUGGING FACE TOKEN
    # ========================================================

    token = os.getenv(
        "HF_TOKEN"
    )

    if not token:

        raise RuntimeError(
            "HF_TOKEN environment variable "
            "is not set."
        )


    # ========================================================
    # CLIENT
    # ========================================================

    client = InferenceClient(
        api_key=token
    )


    # ========================================================
    # SCORE CANDIDATES
    # ========================================================

    results = []


    for _, facet in (
        candidates.iterrows()
    ):

        print(
            f"\nScoring facet: "
            f"{facet['raw_facet']}"
        )


        result = score_facet(
            client,
            conversation,
            facet
        )


        output = build_result(
            facet,
            result
        )


        results.append(
            output
        )


    # ========================================================
    # FINAL PIPELINE OUTPUT
    # ========================================================

    return {

        "conversation":
            conversation,

        "retrieval_model":
            RETRIEVAL_MODEL_NAME,

        "scoring_model":
            SCORING_MODEL_NAME,

        "top_k":
            TOP_K,

        "similarity_threshold":
            SIMILARITY_THRESHOLD,

        "retrieved_count":
            len(retrieved),

        "threshold_pass_count":
            len(thresholded),

        "llm_candidate_limit":
            MAX_LLM_CANDIDATES,

        "candidate_count":
            len(results),

        "results":
            results,
    }


# ============================================================
# LOCAL RETRIEVAL TEST
# ============================================================

def run_local_test(
    conversation
):

    print(
        "\nLOCAL RETRIEVAL TEST MODE"
    )

    print(
        "No Hugging Face inference request "
        "will be made."
    )


    (
        facets,
        retrieved,
        thresholded,
        candidates
    ) = prepare_candidates(
        conversation
    )


    print(
        "\n========================================"
    )

    print(
        "TOP RETRIEVED FACETS"
    )

    print(
        "========================================"
    )


    print(
        retrieved[
            [
                "raw_facet",
                "facet_type",
                "conversation_observable",
                "similarity",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\n========================================"
    )

    print(
        "CANDIDATES FOR LLM"
    )

    print(
        "========================================"
    )


    if candidates.empty:

        print(
            "No candidates passed "
            "the similarity threshold."
        )

    else:

        print(
            candidates[
                [
                    "raw_facet",
                    "facet_type",
                    "conversation_observable",
                    "similarity",
                    "action_match",
                ]
            ].to_string(
                index=False
            )
        )


    print(
        "\n========================================"
    )

    print(
        "LOCAL TEST SUMMARY"
    )

    print(
        "========================================"
    )


    print(
        f"Total facets: "
        f"{len(facets)}"
    )

    print(
        f"Retrieved: "
        f"{len(retrieved)}"
    )

    print(
        f"Threshold: "
        f"{SIMILARITY_THRESHOLD}"
    )

    print(
        f"Passed threshold: "
        f"{len(thresholded)}"
    )

    print(
        f"Maximum LLM candidates: "
        f"{MAX_LLM_CANDIDATES}"
    )

    print(
        f"Selected candidates: "
        f"{len(candidates)}"
    )


    print(
        "\nLOCAL TEST COMPLETE"
    )


# ============================================================
# GET CONVERSATION
# ============================================================

def get_conversation():

    # --------------------------------------------------------
    # Command-line conversation
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        args = [
            arg
            for arg in sys.argv[1:]
            if not arg.startswith("--")
        ]

        if args:

            return " ".join(
                args
            )


    # --------------------------------------------------------
    # Default test conversation
    # --------------------------------------------------------

    return (
        "I kept working on my company for three years "
        "despite repeated failures and setbacks."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    conversation = (
        get_conversation()
    )


    print(
        "\n========================================"
    )

    print(
        "AI FACET SCORING PIPELINE"
    )

    print(
        "========================================"
    )


    print(
        "\nConversation:"
    )

    print(
        conversation
    )


    # ========================================================
    # LOCAL TEST
    # ========================================================

    if "--local-test" in sys.argv:

        run_local_test(
            conversation
        )

        return


    # ========================================================
    # NORMAL PIPELINE
    # ========================================================

    output = run_pipeline(
        conversation
    )


    print(
        "\n========================================"
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "========================================"
    )


    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        )
    )


    print(
        "\n========================================"
    )

    print(
        "PIPELINE COMPLETE"
    )

    print(
        "========================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()