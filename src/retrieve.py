from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FACET_PATH = PROJECT_ROOT / "outputs" / "enriched_facets.csv"
INDEX_PATH = PROJECT_ROOT / "outputs" / "facet_embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 25


def build_index(model, facets):
    embeddings = model.encode(
        facets["normalized_facet"].fillna("").tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    np.save(INDEX_PATH, embeddings)

    return embeddings


def load_or_build_index(model, facets):
    if INDEX_PATH.exists():
        print("Loading cached facet embeddings...")
        return np.load(INDEX_PATH)

    print("Creating facet embeddings...")
    return build_index(model, facets)


def retrieve_facets(
    conversation: str,
    model,
    facets: pd.DataFrame,
    facet_embeddings: np.ndarray
):
    conversation_embedding = model.encode(
        conversation,
        normalize_embeddings=True
    )

    similarities = facet_embeddings @ conversation_embedding

    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    results = facets.iloc[top_indices].copy()
    results["similarity"] = similarities[top_indices]

    return results


def main():

    if not FACET_PATH.exists():
        raise FileNotFoundError(
            f"Facet file not found: {FACET_PATH}"
        )

    facets = pd.read_csv(FACET_PATH)

    model = SentenceTransformer(MODEL_NAME)

    print(f"Loaded {len(facets)} facets.")

    facet_embeddings = load_or_build_index(
        model,
        facets
    )

    conversation = (
        "I left my secure job and started a company "
        "even though I knew I might fail."
    )

    results = retrieve_facets(
        conversation,
        model,
        facets,
        facet_embeddings
    )

    print("\nConversation:")
    print(conversation)

    print(f"\nTop {TOP_K} retrieved facets:")

    print(
        results[
            [
                "raw_facet",
                "facet_type",
                "conversation_observable",
                "similarity"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()