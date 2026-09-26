"""Model-level XAI for the Digital Trust Engine SMS classifier.

Uses Integrated Gradients over DistilBERT input embeddings to estimate which
tokens contributed most to the model's selected prediction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class SMSExplainer:
    """Integrated-Gradients explainer for a Hugging Face text classifier."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.model.eval()

    def explain(
        self,
        text: str,
        target_label: Optional[int] = None,
        steps: int = 20,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Return token-level attribution for one message."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message text cannot be empty.")
        if steps < 2:
            raise ValueError("steps must be at least 2.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        embedding_layer = self.model.get_input_embeddings()

        input_embeddings = embedding_layer(input_ids).detach()
        baseline_embeddings = torch.zeros_like(input_embeddings)

        with torch.no_grad():
            output = self.model(**encoded)
            probabilities = torch.softmax(output.logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        target_id = predicted_id if target_label is None else int(target_label)

        if target_id < 0 or target_id >= output.logits.shape[-1]:
            raise ValueError("target_label is outside the model's class range.")

        total_gradients = torch.zeros_like(input_embeddings)
        delta = input_embeddings - baseline_embeddings

        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            interpolated = (
                baseline_embeddings + alpha * delta
            ).detach().requires_grad_(True)

            step_output = self.model(
                inputs_embeds=interpolated,
                attention_mask=attention_mask,
            )
            target_score = step_output.logits[0, target_id]

            gradients = torch.autograd.grad(
                target_score,
                interpolated,
                retain_graph=False,
                create_graph=False,
            )[0]

            total_gradients += gradients.detach()

        average_gradients = total_gradients / float(steps)
        token_attributions = (delta * average_gradients).sum(dim=-1)[0]

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0].detach().cpu().tolist()
        )
        mask = attention_mask[0].detach().cpu().tolist()

        rows: List[Dict[str, Any]] = []

        # Preserve original token order. The humanizer needs sequence order
        # to reconstruct WordPiece words and URLs correctly.
        for position, (token, attribution, is_active) in enumerate(
            zip(
                tokens,
                token_attributions.detach().cpu().tolist(),
                mask,
            )
        ):
            if not is_active:
                continue
            if token in self.tokenizer.all_special_tokens:
                continue

            rows.append(
                {
                    "position": position,
                    "token": token,
                    "attribution": float(attribution),
                }
            )

        # Keep tokens in message order. Create a separate ranked view for
        # debugging and UI display.
        ranked_rows = sorted(
            rows,
            key=lambda row: abs(row["attribution"]),
            reverse=True,
        )

        return {
            "prediction_id": predicted_id,
            "target_id": target_id,
            "probabilities": probabilities.detach().cpu().tolist(),
            "tokens": rows,
            "ranked_tokens": ranked_rows[:top_k],
            "top_positive": [
                row for row in ranked_rows
                if row["attribution"] > 0
            ][:top_k],
            "top_negative": [
                row for row in sorted(
                    rows,
                    key=lambda row: row["attribution"],
                )
                if row["attribution"] < 0
            ][:top_k],
        }
