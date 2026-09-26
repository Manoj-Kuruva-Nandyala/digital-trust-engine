"""Human-readable XAI utilities for the Digital Trust Engine.

Converts Integrated Gradients token attributions into grouped evidence
that can be used by the senior-citizen-friendly explanation layer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


class HumanReadableXAI:
    """Post-process token-level attributions into useful evidence."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @staticmethod
    def _clean_token(token: str) -> str:
        if token.startswith("##"):
            return token[2:]
        return token

    @staticmethod
    def _is_url_token(token: str) -> bool:
        return token in {":", "/", ".", "-", "_", "?", "=", "&", "%"} or             "http" in token.lower() or "www" in token.lower()

    def group_tokens(self, token_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group WordPiece fragments and adjacent URL/punctuation tokens."""
        groups: List[Dict[str, Any]] = []
        current: Dict[str, Any] | None = None

        for row in token_rows:
            token = row["token"]
            attribution = float(row["attribution"])

            # WordPiece continuation: ##c should join the preceding token.
            if token.startswith("##") and current is not None:
                current["text"] += self._clean_token(token)
                current["attribution"] += attribution
                continue

            is_urlish = self._is_url_token(token)

            if is_urlish and current is not None and current["is_urlish"]:
                current["text"] += token
                current["attribution"] += attribution
                continue

            if current is not None:
                groups.append(current)

            current = {
                "text": self._clean_token(token),
                "attribution": attribution,
                "is_urlish": is_urlish,
            }

        if current is not None:
            groups.append(current)

        return groups

    def extract_evidence(
        self,
        xai_result: Dict[str, Any],
        min_attribution: float = 0.05,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return meaningful positive model evidence.

        URL/punctuation fragments are grouped and shown as one link signal.
        Very short punctuation-only groups are ignored.
        """
        positive_rows = [
            row
            for row in xai_result.get("tokens", [])
            if float(row["attribution"]) >= min_attribution
        ]

        groups = self.group_tokens(positive_rows)

        evidence: List[Dict[str, Any]] = []

        for group in groups:
            text = group["text"].strip()
            attribution = float(group["attribution"])

            if not text:
                continue

            if group["is_urlish"] and re.search(r"https?://|www\.", text, re.I):
                evidence.append({
                    "type": "link",
                    "text": text,
                    "attribution": attribution,
                })
                continue

            # Skip punctuation-only fragments.
            if not re.search(r"[A-Za-z0-9]", text):
                continue

            evidence.append({
                "type": "text",
                "text": text,
                "attribution": attribution,
            })

        evidence.sort(
            key=lambda item: abs(float(item["attribution"])),
            reverse=True,
        )

        return evidence[:top_k]

    @staticmethod
    def to_plain_language(evidence: List[Dict[str, Any]]) -> List[str]:
        """Convert model evidence into simple statements.

        This deliberately does not claim that an individual word proves fraud;
        it says that the model considered the evidence relevant to its
        prediction.
        """
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
            elif any(word in lower for word in ["blocked", "suspended", "suspension", "expire", "expired"]):
                statements.append(
                    f'The wording "{text}" contributed to the risk prediction by warning about a possible account problem.'
                )
            elif any(word in lower for word in ["bank", "account", "kyc", "verify", "verification", "update"]):
                statements.append(
                    f'The wording "{text}" contributed to the risk prediction because it relates to account or identity verification.'
                )
            else:
                statements.append(
                    f'The wording "{text}" contributed to the model risk prediction.'
                )

        return statements
