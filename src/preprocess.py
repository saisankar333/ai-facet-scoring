from pathlib import Path
import re
import pandas as pd


INPUT_PATH = Path("data/Facets Assignment.csv")
OUTPUT_PATH = Path("outputs/enriched_facets.csv")


def normalize_facet(value: str) -> str:
    text = str(value).strip()

    text = re.sub(r"^\s*\d+\.\s*", "", text)
    text = re.sub(r":\s*$", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.lower()


def detect_format_flags(value: str) -> list[str]:
    text = str(value).strip()
    flags = []

    if not text:
        flags.append("empty")

    if re.match(r"^\s*\d+\.\s+", text):
        flags.append("numbered_prefix")

    if text.endswith(":"):
        flags.append("trailing_colon")

    return flags


def classify_facet(facet: str) -> tuple[str, bool, str]:
    """
    Conservative rule-based facet classification.

    Returns:
        facet_type,
        conversation_observable,
        abstention_reason
    """

    text = facet.lower()

    # Medical / laboratory / physical measurements
    medical_keywords = [
        "blood", "glucose", "cholesterol", "blood pressure",
        "heart rate", "medical", "diagnosis", "disease",
        "symptom", "lab", "laboratory", "bmi"
    ]

    # External / biographical facts that normally need verification
    external_keywords = [
        "credit score", "income", "salary", "criminal",
        "employment history", "academic record", "iq test",
        "personality inventory", "hexaco", "enneagram",
        "astrology", "zodiac"
    ]

    # Spiritual/religious practice can be conversationally observable
    spiritual_keywords = [
        "spiritual", "religious", "prayer", "meditation",
        "quran", "bible", "sufi", "hindu", "buddhist",
        "sikh", "kabbalah", "iching", "i ching",
        "bahá", "gnostic", "mantra"
    ]

    # Cognitive / skill-related
    cognitive_keywords = [
        "reasoning", "reasoning subcomponents", "problem solving",
        "numerical", "statistical", "logical", "computer skills",
        "ability", "skills", "decision making", "critical thinking"
    ]

    # Behavioral / interpersonal
    behavioral_keywords = [
        "risk", "assertiveness", "leadership", "persistence",
        "hesitation", "cunning", "adventure", "overprotectiveness",
        "honesty", "humility", "empathy", "communication",
        "relationship", "affiliation", "motivation", "achievement",
        "adaptability", "flexibility", "work styles", "behavior",
        "behaviour", "conscientiousness", "emotionalism",
        "discontentment", "merriness", "self-improvement"
    ]

    if any(keyword in text for keyword in medical_keywords):
        return (
            "medical_or_physical",
            False,
            "Requires medical, laboratory, or physical evidence."
        )

    if any(keyword in text for keyword in external_keywords):
        return (
            "external_or_biographical",
            False,
            "Requires external or independently verified information."
        )

    if any(keyword in text for keyword in spiritual_keywords):
        return (
            "spiritual_or_religious",
            True,
            "Abstain if the conversation does not provide direct evidence."
        )

    if any(keyword in text for keyword in cognitive_keywords):
        return (
            "cognitive_or_skill",
            True,
            "Abstain if the conversation does not provide enough evidence."
        )

    if any(keyword in text for keyword in behavioral_keywords):
        return (
            "behavioral_or_interpersonal",
            True,
            "Abstain if the conversation does not provide enough evidence."
        )

    # Conservative default:
    # potentially observable, but only score with sufficient evidence.
    return (
        "other_or_uncertain",
        True,
        "Abstain when the conversation does not provide sufficient evidence."
    )


def scoring_definition(facet_type: str) -> str:
    if facet_type == "medical_or_physical":
        return "Do not directly score from ordinary conversation; abstain unless valid evidence is explicitly available."

    if facet_type == "external_or_biographical":
        return "Score only when the conversation itself provides sufficient direct evidence; otherwise abstain."

    return "Score the strength of conversational evidence on an ordered five-level scale."


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    if "Facets" not in df.columns:
        raise ValueError(
            "Expected a 'Facets' column in the input CSV."
        )

    enriched = pd.DataFrame()

    # Preserve original source value.
    enriched["raw_facet"] = df["Facets"]

    # Create normalized representation.
    enriched["normalized_facet"] = (
        df["Facets"]
        .fillna("")
        .apply(normalize_facet)
    )

    # Detect formatting anomalies.
    enriched["format_flags"] = (
        df["Facets"]
        .fillna("")
        .apply(detect_format_flags)
        .apply(lambda flags: "|".join(flags))
    )

    # Semantic enrichment.
    classifications = enriched["normalized_facet"].apply(classify_facet)

    enriched["facet_type"] = classifications.apply(lambda x: x[0])
    enriched["conversation_observable"] = classifications.apply(lambda x: x[1])
    enriched["abstention_reason"] = classifications.apply(lambda x: x[2])

    enriched["sensitivity"] = enriched["facet_type"].apply(
        lambda x: "high"
        if x == "medical_or_physical"
        else "medium"
        if x == "external_or_biographical"
        else "low"
    )

    enriched["scoring_definition"] = enriched["facet_type"].apply(
        scoring_definition
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    enriched.to_csv(OUTPUT_PATH, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(enriched)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFacet type counts:")
    print(enriched["facet_type"].value_counts().to_string())

    print("\nConversation observable:")
    print(
        enriched["conversation_observable"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()