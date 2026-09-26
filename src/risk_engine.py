"""Risk scoring utilities for the Digital Trust Engine.

The engine converts model class probabilities into a user-facing fraud
probability, fraud score, and risk level. It does not assign a score
independently of the model prediction.
"""

from typing import Dict, Any


class FraudRiskEngine:
    """Convert SMS/email model probabilities into fraud risk information."""

    def __init__(
        self,
        fraud_labels=("spam", "smishing"),
        high_threshold=70.0,
        medium_threshold=30.0,
    ):
        self.fraud_labels = tuple(label.lower() for label in fraud_labels)
        self.high_threshold = float(high_threshold)
        self.medium_threshold = float(medium_threshold)

    def calculate_fraud_probability(self, probabilities: Dict[str, float]) -> float:
        """Return fraud probability as the sum of configured fraud classes."""
        normalized = {
            str(label).lower(): float(probability)
            for label, probability in probabilities.items()
        }

        missing = [label for label in self.fraud_labels if label not in normalized]
        if missing:
            raise ValueError(
                f"Missing probability for required fraud label(s): {missing}"
            )

        fraud_probability = sum(normalized[label] for label in self.fraud_labels)
        return max(0.0, min(1.0, fraud_probability))

    def calculate_fraud_score(self, probabilities: Dict[str, float]) -> float:
        """Return fraud probability on a 0-100 scale."""
        return round(self.calculate_fraud_probability(probabilities) * 100, 2)

    def risk_level(self, fraud_score: float) -> str:
        """Map fraud score to a simple risk level."""
        if fraud_score >= self.high_threshold:
            return "HIGH RISK"
        if fraud_score >= self.medium_threshold:
            return "MEDIUM RISK"
        return "LOW RISK"

    def analyze(self, prediction: str, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """Return the complete model-driven risk result."""
        fraud_probability = self.calculate_fraud_probability(probabilities)
        fraud_score = round(fraud_probability * 100, 2)

        return {
            "prediction": prediction,
            "probabilities": dict(probabilities),
            "fraud_probability": round(fraud_probability, 4),
            "fraud_score": fraud_score,
            "risk_level": self.risk_level(fraud_score),
        }
