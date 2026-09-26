"""SMS model utilities for the Digital Trust Engine.

Loads the trained DistilBERT SMS classifier saved in Google Drive and
provides a small reusable prediction function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_LABELS = {
    0: "ham",
    1: "spam",
    2: "smishing",
}


class SMSModel:
    """Reusable wrapper around the trained DistilBERT SMS classifier."""

    def __init__(
        self,
        model_path: str,
        id_to_label: Dict[int, str] | None = None,
    ) -> None:
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"SMS model directory not found: {model_dir}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        self.id_to_label = id_to_label or DEFAULT_LABELS

    def predict(self, text: str) -> Dict[str, Any]:
        """Predict the class and probabilities for one SMS/message."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message text cannot be empty.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits, dim=-1
        )[0].detach().cpu().numpy()

        predicted_id = int(np.argmax(probabilities))

        return {
            "prediction": self.id_to_label[predicted_id],
            "probabilities": {
                self.id_to_label[i]: float(probabilities[i])
                for i in range(len(probabilities))
            },
        }
