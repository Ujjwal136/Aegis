from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config import settings


@dataclass
class RedactionResult:
    redacted_text: str
    redactions: list[str]


class Redactor:
    def __init__(self) -> None:
        self.ner_pipeline = None
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
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(settings.redactor_model_path)
            model = AutoModelForTokenClassification.from_pretrained(settings.redactor_model_path)
            self.ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            return True
        except Exception:
            self.ner_pipeline = None
            return False

    def redact(self, text: str) -> dict[str, Any]:
        redacted = text
        redactions: list[str] = []

        if self.ner_pipeline is not None:
            try:
                entities = self.ner_pipeline(text)
                for ent in entities:
                    entity_word = ent.get("word", "")
                    entity_group = str(ent.get("entity_group", "PII")).upper()
                    if not entity_word:
                        continue
                    token = f"[{entity_group}_REDACTED]"
                    if entity_word in redacted:
                        redacted = redacted.replace(entity_word, token)
                        redactions.append(entity_group)
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
