"""Email DistilBERT inference module for the Digital Trust Engine.

Loads the dedicated email classifier trained on the phishing-email dataset.
The model files are kept outside GitHub (for example in Google Drive); pass
their local path when creating EmailModel.
"""

import os
from typing import Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class EmailModel:
    """Analyze email subject and body with the dedicated Email DistilBERT."""

    def __init__(self, model_path: str | None = None, max_length: int = 256):
        model_path = model_path or os.getenv("DIGITAL_TRUST_EMAIL_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "Email model path is required. Pass model_path or set "
                "DIGITAL_TRUST_EMAIL_MODEL_PATH."
            )

        self.model_path = model_path
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.id_to_label = {
            int(k): v for k, v in self.model.config.id2label.items()
        }

    @staticmethod
    def combine_text(subject: str = "", body: str = "") -> str:
        """Combine email subject and body while preserving their semantic roles."""
        subject = (subject or "").strip()
        body = (body or "").strip()

        if subject and body:
            return f"Subject: {subject}\n\n{body}"
        return subject or body

    def predict(self, subject: str = "", body: str = "") -> Dict[str, Any]:
        """Return the email prediction and class probabilities."""
        text = self.combine_text(subject, body)

        if not text:
            raise ValueError("Email subject or body must contain text.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        predicted_id = int(np.argmax(probabilities))

        return {
            "input_type": "email",
            "subject": subject,
            "body": body,
            "text": text,
            "prediction": self.id_to_label[predicted_id],
            "probabilities": {
                self.id_to_label[i]: float(probabilities[i])
                for i in range(len(probabilities))
            },
        }
