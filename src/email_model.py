"""Prototype email analyzer for the Digital Trust Engine.

The current prototype reuses the trained SMS DistilBERT text classifier for
email subject/body text. This is intentionally kept separate from the SMS
module so that a dedicated email model and email-specific training data can
be introduced later without changing the rest of the pipeline.
"""

from typing import Dict, Any


class EmailModel:
    """Analyze email subject and body using the existing text classifier."""

    def __init__(self, sms_model):
        self.model = sms_model

    @staticmethod
    def combine_text(subject: str = "", body: str = "") -> str:
        """Combine subject and body while preserving their semantic roles."""
        subject = (subject or "").strip()
        body = (body or "").strip()

        if subject and body:
            return f"Subject: {subject}\n\n{body}"
        return subject or body

    def predict(self, subject: str = "", body: str = "") -> Dict[str, Any]:
        """Return the model prediction and probabilities for an email."""
        text = self.combine_text(subject, body)

        if not text:
            raise ValueError("Email subject or body must contain text.")

        result = self.model.predict(text)

        return {
            "input_type": "email",
            "subject": subject,
            "body": body,
            "text": text,
            "prediction": result["prediction"],
            "probabilities": result["probabilities"],
        }
