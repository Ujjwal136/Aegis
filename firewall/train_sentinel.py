from __future__ import annotations

import csv
import math
import pickle
import re
from collections import Counter
from pathlib import Path


TOKEN_RE = re.compile(r"\b\w+\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_rows(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = (row.get("prompt") or "").strip()
            label = int(row.get("label") or 0)
            if prompt:
                rows.append((prompt, label))
    return rows


def train_naive_bayes(rows: list[tuple[str, int]]) -> dict:
    attack_counter = Counter()
    clean_counter = Counter()
    attack_docs = 0
    clean_docs = 0

    for prompt, label in rows:
        tokens = tokenize(prompt)
        if label == 1:
            attack_docs += 1
            attack_counter.update(tokens)
        else:
            clean_docs += 1
            clean_counter.update(tokens)

    vocab = sorted(set(attack_counter) | set(clean_counter))
    attack_total = sum(attack_counter.values())
    clean_total = sum(clean_counter.values())
    v = max(len(vocab), 1)

    token_scores: dict[str, float] = {}
    for token in vocab:
        p_attack = (attack_counter[token] + 1.0) / (attack_total + v)
        p_clean = (clean_counter[token] + 1.0) / (clean_total + v)
        token_scores[token] = math.log(p_attack / p_clean)

    prior = math.log((attack_docs + 1.0) / (clean_docs + 1.0))

    return {
        "prior": prior,
        "token_scores": token_scores,
    }


def predict(model: dict, text: str) -> tuple[int, float]:
    score = model["prior"]
    for token in tokenize(text):
        score += model["token_scores"].get(token, 0.0)
    prob_attack = 1.0 / (1.0 + math.exp(-max(min(score, 50), -50)))
    return (1 if prob_attack >= 0.5 else 0, prob_attack)


def evaluate(model: dict, rows: list[tuple[str, int]]) -> tuple[float, dict[str, int]]:
    tp = tn = fp = fn = 0
    for prompt, label in rows:
        pred, _ = predict(model, prompt)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    accuracy = (tp + tn) / max(len(rows), 1)
    return accuracy, {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main() -> None:
    datasets = [Path("aegis_dataset.csv"), Path("aegis_hybrid_dataset.csv")]
    rows: list[tuple[str, int]] = []
    for ds in datasets:
        if ds.exists():
            rows.extend(load_rows(ds))

    if not rows:
        raise FileNotFoundError("No training rows found in local datasets.")

    split_index = int(len(rows) * 0.8)
    train_rows = rows[:split_index]
    test_rows = rows[split_index:]

    model = train_naive_bayes(train_rows)
    accuracy, cm = evaluate(model, test_rows)

    with open("sentinel_model.joblib", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.joblib", "wb") as f:
        pickle.dump({"tokenizer": "regex_word"}, f)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Confusion Matrix: TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")
    print("Saved: sentinel_model.joblib, vectorizer.joblib")


if __name__ == "__main__":
    main()
