"""Human-readable XAI utilities for the Digital Trust Engine.

Reconstructs WordPiece tokens before selecting evidence, preserves complete
URLs as one evidence item, and filters punctuation/common low-information
tokens without changing the underlying model attributions.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


_COMMON_LOW_VALUE = {
    "a", "an", "the", "be", "is", "are", "was", "were", "will", "would",
    "could", "should", "to", "of", "in", "on", "at", "for", "and", "or",
    "you", "your", "this", "that", "it", "we", "our", "i",
}

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s]+", re.IGNORECASE)


class HumanReadableXAI:
    """Post-process token-level attributions into useful evidence."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def reconstruct_words(self, token_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reconstruct WordPiece tokens while preserving attribution sums."""
        groups: List[Dict[str, Any]] = []
        current: Dict[str, Any] | None = None

        for row in token_rows:
            token = str(row["token"])
            attribution = float(row["attribution"])

            if token.startswith("##") and current is not None:
                current["text"] += token[2:]
                current["attribution"] += attribution
                continue

            if current is not None:
                groups.append(current)

            current = {"text": token, "attribution": attribution}

        if current is not None:
            groups.append(current)

        return groups

    @staticmethod
    def _normalise_word(text: str) -> str:
        return text.replace("##", "").strip()

    @staticmethod
    def _is_meaningful(text: str) -> bool:
        cleaned = text.strip().lower()
        if not cleaned:
            return False
        if cleaned in _COMMON_LOW_VALUE:
            return False
        if not re.search(r"[a-z0-9]", cleaned):
            return False
        return True

    def extract_evidence(
        self,
        xai_result: Dict[str, Any],
        min_attribution: float = 0.05,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return meaningful positive model evidence."""
        rows = [
            row for row in xai_result.get("tokens", [])
            if float(row["attribution"]) >= min_attribution
        ]

        groups = self.reconstruct_words(rows)

        readable = " ".join(
            self._normalise_word(group["text"]) for group in groups
        )

        url_match = _URL_RE.search(readable)
        evidence: List[Dict[str, Any]] = []

        if url_match:
            url_text = url_match.group(0).rstrip(".,!?;")
            url_attribution = sum(
                float(group["attribution"])
                for group in groups
                if group["text"] in url_text
            )
            evidence.append({
                "type": "link",
                "text": url_text,
                "attribution": url_attribution,
            })

        for group in groups:
            text = self._normalise_word(group["text"])
            lower = text.lower()

            if not self._is_meaningful(text):
                continue

            if url_match and text in url_match.group(0):
                continue

            evidence.append({
                "type": "text",
                "text": text,
                "attribution": float(group["attribution"]),
            })

        evidence.sort(
            key=lambda item: abs(float(item["attribution"])),
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
        """Convert model evidence into senior-friendly statements."""
        statements: List[str] = []

        for item in evidence:
            text = item["text"].strip()
            lower = text.lower()

            if item["type"] == "link":
                statements.append(
                    "The message contains a link that contributed to the risk prediction."
                )
            elif any(word in lower for word in ["urgent", "immediately", "now", "today"]):
                statements.append(
                    f'The wording "{text}" contributed to the risk prediction by creating a sense of urgency.'
                )
            elif any(word in lower for word in [
                "blocked", "suspended", "suspension", "expire", "expired"
            ]):
                statements.append(
                    f'The wording "{text}" contributed to the risk prediction by warning about a possible account problem.'
                )
            elif any(word in lower for word in [
                "bank", "account", "kyc", "verify", "verification", "update"
            ]):
                statements.append(
                    f'The wording "{text}" contributed to the risk prediction because it relates to account or identity verification.'
                )
            else:
                statements.append(
                    f'The wording "{text}" contributed to the model risk prediction.'
                )

        return statements
