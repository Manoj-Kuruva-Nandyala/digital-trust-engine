"""Human-readable XAI utilities for the Digital Trust Engine.

Reconstructs the original token sequence before filtering and aggregates
WordPiece attributions correctly. URLs are represented as a single link
signal, without claiming that individual domain words caused the prediction.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


_COMMON_LOW_VALUE = {
    "a", "an", "the", "be", "is", "are", "was", "were", "will", "would",
    "could", "should", "to", "of", "in", "on", "at", "for", "and", "or",
    "you", "your", "this", "that", "it", "we", "our", "i",
}

_URL_START = {"http", "https", "www"}


class HumanReadableXAI:
    """Convert token attributions into human-readable model evidence."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def reconstruct(self, token_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reconstruct WordPiece words and URL spans in original order."""
        groups: List[Dict[str, Any]] = []
        current_word: Dict[str, Any] | None = None
        current_url: Dict[str, Any] | None = None
        in_url = False

        for row in token_rows:
            token = str(row["token"])
            attribution = float(row["attribution"])

            if in_url:
                # Continue collecting URL punctuation/text until the URL span ends.
                if (
                    token
                    or re.match(r"[A-Za-z0-9./:?=&%_-]", token)
                ):
                    current_url["text"] += token
                    current_url["attribution"] += attribution
                    continue

            if token.lower() in _URL_START:
                if current_word is not None:
                    groups.append(current_word)
                    current_word = None
                current_url = {
                    "type": "link",
                    "text": token,
                    "attribution": attribution,
                }
                groups.append(current_url)
                in_url = True
                continue

            if in_url:
                in_url = False
                current_url = None

            if token.startswith("##") and current_word is not None:
                current_word["text"] += token[2:]
                current_word["attribution"] += attribution
                continue

            if current_word is not None:
                groups.append(current_word)

            current_word = {
                "type": "text",
                "text": token,
                "attribution": attribution,
            }

        if current_word is not None:
            groups.append(current_word)

        return groups

    @staticmethod
    def _is_meaningful_text(text: str) -> bool:
        value = text.strip().lower()
        if not value or value in _COMMON_LOW_VALUE:
            return False
        return bool(re.search(r"[a-z0-9]", value))

    def extract_evidence(
        self,
        xai_result: Dict[str, Any],
        min_attribution: float = 0.05,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return positive model evidence after correct reconstruction."""
        groups = self.reconstruct(xai_result.get("tokens", []))
        evidence: List[Dict[str, Any]] = []

        for group in groups:
            text = group["text"].strip()
            attribution = float(group["attribution"])

            if group["type"] == "link":
                # A link is evidence only when its aggregate attribution is
                # positive. We intentionally do not interpret domain words.
                if attribution > 0:
                    evidence.append({
                        "type": "link",
                        "text": text,
                        "attribution": attribution,
                    })
                continue

            if attribution < min_attribution:
                continue

            if not self._is_meaningful_text(text):
                continue

            evidence.append({
                "type": "text",
                "text": text,
                "attribution": attribution,
            })

        evidence.sort(
            key=lambda item: float(item["attribution"]),
            reverse=True,
        )

        unique: Dict[str, Dict[str, Any]] = {}
        for item in evidence:
            key = item["text"].lower()
            if key not in unique:
                unique[key] = item

        return list(unique.values())[:top_k]

    @staticmethod
    def to_plain_language(evidence: List[Dict[str, Any]]) -> List[str]:
        """Translate model evidence into senior-friendly statements."""
        statements: List[str] = []

        for item in evidence:
            text = item["text"].strip()
            lower = text.lower()

            if item["type"] == "link":
                statements.append(
                    "The message contains a web link that contributed to the risk prediction."
                )
            elif any(
                word in lower
                for word in ["urgent", "immediately", "now", "today"]
            ):
                statements.append(
                    f'The wording "{text}" contributed to the risk prediction by creating a sense of urgency.'
                )
            elif any(
                word in lower
                for word in [
                    "blocked", "suspended", "suspension",
                    "expire", "expired",
                ]
            ):
                statements.append(
                    f'The wording "{text}" contributed to the risk prediction by warning about a possible account problem.'
                )
            elif any(
                word in lower
                for word in [
                    "bank", "account", "kyc", "verify",
                    "verification", "update",
                ]
            ):
                statements.append(
                    f'The wording "{text}" contributed to the risk prediction because it relates to account or identity verification.'
                )
            else:
                statements.append(
                    f'The wording "{text}" contributed to the model risk prediction.'
                )

        return statements
