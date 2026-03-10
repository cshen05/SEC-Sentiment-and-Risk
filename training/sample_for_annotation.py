

import csv
import random
import sys
from pathlib import Path

# Allow running as: python3 -m training.sample_for_annotation
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.db import get_connection

OUTPUT_FILE = PROJECT_ROOT / "data" / "annotation_sample.csv"
TOTAL_SAMPLE_SIZE = 200
RANDOM_SEED = 42


def fetch_rows_by_label(cur, label: str):
    cur.execute(
        """
        SELECT id, text, weak_label
        FROM paragraphs
        WHERE weak_label = %s
        ORDER BY id
        """,
        (label,),
    )
    return cur.fetchall()


def main() -> None:
    random.seed(RANDOM_SEED)

    conn = get_connection()
    cur = conn.cursor()

    try:
        negative_rows = fetch_rows_by_label(cur, "negative_risk")
        positive_rows = fetch_rows_by_label(cur, "positive_outlook")
        neutral_rows = fetch_rows_by_label(cur, "neutral")
    finally:
        cur.close()
        conn.close()

    print(f"[INFO] negative_risk rows available: {len(negative_rows)}")
    print(f"[INFO] positive_outlook rows available: {len(positive_rows)}")
    print(f"[INFO] neutral rows available: {len(neutral_rows)}")

    selected_rows = []

    # Keep all minority-class rows so the evaluation set is useful.
    selected_rows.extend(positive_rows)
    selected_rows.extend(neutral_rows)

    remaining_slots = TOTAL_SAMPLE_SIZE - len(selected_rows)
    if remaining_slots < 0:
        raise ValueError(
            "Minority-class rows exceed TOTAL_SAMPLE_SIZE. Increase TOTAL_SAMPLE_SIZE."
        )

    if len(negative_rows) < remaining_slots:
        raise ValueError(
            "Not enough negative_risk rows to complete the requested sample size."
        )

    selected_rows.extend(random.sample(negative_rows, remaining_slots))
    random.shuffle(selected_rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "weak_label", "manual_label"])
        for paragraph_id, text, weak_label in selected_rows:
            writer.writerow([paragraph_id, text, weak_label, ""])

    print(f"[INFO] Wrote {len(selected_rows)} rows to {OUTPUT_FILE}")
    print("[INFO] Sample composition:")
    print(f"        negative_risk: {sum(1 for _, _, label in selected_rows if label == 'negative_risk')}")
    print(f"        positive_outlook: {sum(1 for _, _, label in selected_rows if label == 'positive_outlook')}")
    print(f"        neutral: {sum(1 for _, _, label in selected_rows if label == 'neutral')}")


if __name__ == "__main__":
    main()