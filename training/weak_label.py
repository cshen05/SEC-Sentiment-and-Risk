

import sys
from pathlib import Path

import pysentiment2 as ps

# Allow running as: python3 -m training.weak_label
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db import get_connection


lm = ps.LM()


def clean_text(text: str) -> str:
    """Fix common mojibake patterns from SEC text extraction."""
    replacements = {
        "Äôs": "'s",
        "Äô": "'",
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "Â": "",
    }

    cleaned = text
    for bad, good in replacements.items():
        cleaned = cleaned.replace(bad, good)

    return cleaned


def weak_label_paragraph(text: str) -> str:
    """Assign a weak sentiment label using the Loughran-McDonald dictionary."""
    text = clean_text(text)
    tokens = lm.tokenize(text)
    score = lm.get_score(tokens)
    polarity = score["Polarity"]

    if polarity < -0.1:
        return "negative_risk"
    if polarity > 0.1:
        return "positive_outlook"
    return "neutral"


def main() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, text
        FROM paragraphs
        ORDER BY id
        """
    )
    rows = cur.fetchall()

    print(f"[INFO] Found {len(rows)} paragraphs to weak-label")

    updated = 0

    try:
        for paragraph_id, text in rows:
            label = weak_label_paragraph(text)

            cur.execute(
                """
                UPDATE paragraphs
                SET weak_label = %s
                WHERE id = %s
                """,
                (label, paragraph_id),
            )
            updated += 1

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    print(f"[INFO] Updated weak labels for {updated} paragraphs")


if __name__ == "__main__":
    main()