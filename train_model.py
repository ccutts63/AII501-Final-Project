"""Train and evaluate RECCALL's independently trained Bloom's Taxonomy classifier.

Run once before starting the Streamlit app:
    python train_model.py

Dataset: mouryat9/CogBench (Hugging Face), CC BY 4.0.
Dataset revision pinned for reproducibility. Downloaded data is cached under data/.
The 739 human-annotated questions are excluded from training, including
any training examples with the same normalized question text.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REVISION = "2b6b6a2f1292eb11455d4a316b68b4a1c1bf2b32"
DATASET_URL = "https://huggingface.co/datasets/mouryat9/CogBench"
LABELS = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
FILES = ["train.jsonl", "test.jsonl", "human_annotated.jsonl"]


def download_dataset(name: str) -> Path:
    """Download public dataset split when it is not already cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / name
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"Using cached {destination}")
        return destination

    url = f"{DATASET_URL}/resolve/{REVISION}/data/{name}"
    print(f"Downloading {name} from CogBench...")
    req = Request(url, headers={"User-Agent": "RECCALL-AII501/1.0"})
    try:
        with urlopen(req, timeout=90) as response:
            with destination.with_suffix(".part").open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        destination.with_suffix(".part").replace(destination)
    except Exception as exc:
        destination.with_suffix(".part").unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {name}. Check your internet connection, or manually "
            f"download the file from {url} into {DATA_DIR}/, then rerun training."
        ) from exc
    return destination


def load_questions(path: Path) -> list[dict]:
    """Read the original labeled question records; fail on malformed data."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question = str(item.get("question_text") or "").strip()
            label = item.get("bloom_name")
            if question and label in LABELS:
                records.append({"question": question, "label": label})
            else:
                raise ValueError(f"Invalid question/label in {path.name}, line {line_number}")
    if not records:
        raise ValueError(f"No valid records found in {path}")
    return records


def normalized_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.casefold()).strip()


def unique_except(records: list[dict], excluded: set[str]) -> list[dict]:
    """Avoid identical questions across train and held-out evaluation sets."""
    seen = set(excluded)
    result = []
    for item in records:
        key = normalized_question(item["question"])
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def calculate_metrics(model: Pipeline, records: list[dict]) -> dict:
    y_true = [item["label"] for item in records]
    y_pred = model.predict([item["question"] for item in records]).tolist()
    return {
        "samples": len(records),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def train() -> dict:
    for name in FILES:
        download_dataset(name)

    raw_train = load_questions(DATA_DIR / "train.jsonl")
    raw_test = load_questions(DATA_DIR / "test.jsonl")
    raw_gold = load_questions(DATA_DIR / "human_annotated.jsonl")

    # Remove duplicates from the gold set, then EXCLUDE all gold questions
    # from training. This protects the independently evaluated gold score.
    gold = unique_except(raw_gold, set())
    gold_keys = {normalized_question(item["question"]) for item in gold}
    training = unique_except(raw_train, gold_keys)
    train_keys = {normalized_question(item["question"]) for item in training}

    # The official 'test' split is mostly silver/automatically labeled.
    # Exclude any duplicate questions shared with training or gold.
    test = unique_except(raw_test, train_keys | gold_keys)
    if not training or not test or not gold:
        raise RuntimeError("One of the train/test/human-labeled splits is empty after deduplication")
    distribution = Counter(item["label"] for item in training)
    missing = set(LABELS) - set(distribution)
    if missing:
        raise RuntimeError(f"Training data is missing Bloom levels: {', '.join(sorted(missing))}")

    print(f"Training on {len(training):,} questions ({dict(distribution)})")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=2, max_features=25000, sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1200, class_weight="balanced", random_state=42,
        )),
    ])
    model.fit([item["question"] for item in training], [item["label"] for item in training])

    results = {
        "dataset": DATASET_URL,
        "dataset_revision": REVISION,
        "dataset_license": "CC BY 4.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "TF-IDF (word unigrams/bigrams) + Logistic Regression (class-weighted)",
        "train_samples": len(training),
        "train_label_distribution": dict(distribution),
        "labels": LABELS,
        "gold_evaluation": calculate_metrics(model, gold),
        "silver_test_evaluation": calculate_metrics(model, test),
        "notes": (
            "Gold questions are held out of training by normalized text; the official test "
            "split has mostly automatically assigned silver labels. Bloom labels describe "
            "cognitive tasks, not a validated measurement of question difficulty."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "bloom_classifier.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report_lines = [
        "RECCALL — Bloom's Taxonomy classifier evaluation",
        "=" * 55,
        f"Dataset: {DATASET_URL}",
        f"Dataset revision: {REVISION}",
        f"Training examples: {len(training):,}",
        "Model: TF-IDF + Logistic Regression",
        "",
    ]
    for key, heading in [
        ("gold_evaluation", "HELD-OUT HUMAN-LABELLED EVALUATION"),
        ("silver_test_evaluation", "OFFICIAL SILVER-LABELLED TEST SPLIT"),
    ]:
        result = results[key]
        report_lines.extend([
            heading,
            f"Test questions: {result['samples']:,}",
            f"Accuracy: {result['accuracy']:.2%}",
            f"Macro F1: {result['macro_f1']:.3f}",
            "Per-level precision / recall / F1 / support:",
        ])
        for level in LABELS:
            entry = result["classification_report"][level]
            report_lines.append(
                f"  {level:<11} precision={entry['precision']:.3f} "
                f"recall={entry['recall']:.3f} F1={entry['f1-score']:.3f} "
                f"n={int(entry['support'])}"
            )
        report_lines.append("")
    report_lines.append(results["notes"])
    (MODEL_DIR / "evaluation.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(report_lines))
    print(f"\nSaved trained model and evaluation to: {MODEL_DIR}")
    return results


if __name__ == "__main__":
    train()
