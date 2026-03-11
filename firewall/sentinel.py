from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass

from config import settings


THREAT_PATTERNS = {
    "prompt_injection": re.compile(r"ignore|override|system prompt|disable_redaction|do anything now", re.IGNORECASE),
    "data_exfiltration": re.compile(r"aadhaar|pan|upi|passport|dump|all records|unmasked|raw", re.IGNORECASE),
    "sql_injection": re.compile(r"\bunion\b|\bdrop\b|\bor\s+'1'='1|;\s*select", re.IGNORECASE),
}

TOKEN_RE = re.compile(r"\b\w+\b", re.IGNORECASE)


@dataclass
class SentinelResult:
    is_threat: bool
    confidence: float
    threat_type: str


class Sentinel:
    def __init__(self) -> None:
        self.model: dict | None = None

    def load(self) -> bool:
        try:
            with open(settings.sentinel_model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        except FileNotFoundError:
            self.model = None
            return False

    def scan(self, prompt: str) -> dict:
        heuristic_hit = self._threat_type(prompt)
        if heuristic_hit == "none" and self._is_safe_banking_intent(prompt):
            return {
                "is_threat": False,
                "confidence": 0.05,
                "threat_type": "none",
            }

        if self.model:
            is_threat, confidence = self._model_predict(prompt)
            # Block on model only at high confidence; regex hits always block.
            if heuristic_hit != "none":
                is_threat = True
                threat_type = heuristic_hit
            else:
                is_threat = is_threat and confidence >= 0.85
                threat_type = "model_risk" if is_threat else "none"
            return {
                "is_threat": is_threat,
                "confidence": confidence,
                "threat_type": threat_type,
            }

        return {
            "is_threat": heuristic_hit != "none",
            "confidence": 0.9 if heuristic_hit != "none" else 0.1,
            "threat_type": heuristic_hit,
        }

    def _is_safe_banking_intent(self, prompt: str) -> bool:
        lowered = prompt.lower()
        safe_terms = ["balance", "transaction", "loan", "ifsc", "interest", "customer", "cust"]
        return any(term in lowered for term in safe_terms)

    def _model_predict(self, prompt: str) -> tuple[bool, float]:
        score = self.model.get("prior", 0.0)
        token_scores = self.model.get("token_scores", {})
        for token in TOKEN_RE.findall(prompt.lower()):
            score += token_scores.get(token, 0.0)
        prob_attack = 1.0 / (1.0 + math.exp(-max(min(score, 50), -50)))
        return (prob_attack >= 0.55, float(prob_attack))

    def _threat_type(self, prompt: str) -> str:
        for name, pattern in THREAT_PATTERNS.items():
            if pattern.search(prompt):
                return name
        return "none"
