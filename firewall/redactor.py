from __future__ import annotations

import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from config import settings
from firewall.train_redactor import extract_features


@dataclass
class RedactionResult:
    redacted_text: str
    redactions: list[str]


class Redactor:
    def __init__(self) -> None:
        self.ner_model: dict | None = None
        self.ner_classes: set[str] = set()
        self._compiled_patterns = {
            "AADHAAR": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
            "PAN": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE),
            "IFSC": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE),
            "EMAIL": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"),
            "UPI": re.compile(r"\b[a-zA-Z0-9._-]{2,}@(?!.*\.)[a-zA-Z]{2,}\b"),
            "PHONE": re.compile(r"(?:\+91[-\s]?)?[6-9]\d{9}\b"),
            "DOB": re.compile(r"\b(?:\d{2}[/-]\d{2}[/-]\d{4}|\d{4}-\d{2}-\d{2})\b"),
            "PASSPORT": re.compile(r"\b[A-Z][0-9]{7}\b", re.IGNORECASE),
            "ACCOUNT_NO": re.compile(r"\b\d{11,16}\b"),
        }

    def load(self) -> bool:
        try:
            with open(settings.ner_model_path, "rb") as f:
                data = pickle.load(f)
            self.ner_model = data["weights"]
            self.ner_classes = set(data["classes"])
            return True
        except FileNotFoundError:
            self.ner_model = None
            return False

    def _ner_predict(self, tokens: list[str]) -> list[str]:
        """Run the Averaged Perceptron NER tagger over tokens."""
        labels = []
        prev_label = "O"
        weights = self.ner_model
        classes = self.ner_classes
        for i in range(len(tokens)):
            features = extract_features(tokens, i, prev_label)
            scores: dict[str, float] = defaultdict(float)
            for feat, value in features.items():
                if feat in weights:
                    for label, weight in weights[feat].items():
                        scores[label] += value * weight
            label = max(classes, key=lambda c: scores.get(c, 0.0))
            labels.append(label)
            prev_label = label
        return labels

    def redact(self, text: str) -> dict[str, Any]:
        redacted = text
        redactions: list[str] = []

        # NER model pass: detect entities and redact them
        if self.ner_model is not None:
            try:
                tokens = re.findall(r"\S+", redacted)
                if tokens:
                    labels = self._ner_predict(tokens)
                    # Build entity spans (right-to-left replacement to preserve indices)
                    spans: list[tuple[int, int, str]] = []
                    for token, label in zip(tokens, labels):
                        if label.startswith("B-") or label.startswith("I-"):
                            etype = label[2:]
                            start = redacted.find(token)
                            if start != -1:
                                spans.append((start, start + len(token), etype))
                    # Replace right-to-left so indices stay valid
                    for start, end, etype in sorted(spans, key=lambda s: s[0], reverse=True):
                        tag = f"[{etype}_REDACTED]"
                        redacted = redacted[:start] + tag + redacted[end:]
                        redactions.append(etype)
            except Exception:
                pass

        # Regex fallback stays active regardless of model state for stronger safety.
        for pii_type, pattern in self._compiled_patterns.items():
            if pattern.search(redacted):
                redacted = pattern.sub(f"[{pii_type}_REDACTED]", redacted)
                redactions.append(pii_type)

        unique_redactions = sorted(set(redactions))
        return {
            "redacted_text": redacted,
            "redactions": unique_redactions,
        }
