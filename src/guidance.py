"""Senior-friendly actionable guidance for the Digital Trust Engine."""

from typing import Dict, List


class ActionGuidance:
    """Generate plain-language safety guidance from fraud risk results."""

    def get_guidance(self, risk_result: Dict) -> List[str]:
        """Return actionable guidance appropriate to the calculated risk level."""
        risk_level = str(risk_result.get("risk_level", "")).upper()

        if risk_level == "HIGH RISK":
            return [
                "Do not click the link or reply to this message.",
                "Do not share your OTP, password, or bank details.",
                "Check with your bank using its official app or website.",
            ]

        if risk_level == "MEDIUM RISK":
            return [
                "Do not click links or share personal or financial information yet.",
                "Verify the request through the organization's official app or website.",
                "If you are unsure, contact the organization using a trusted phone number.",
            ]

        return [
            "You can continue normally, but always be careful with unexpected requests.",
            "Never share your OTP, password, or bank details because someone asks for them unexpectedly.",
        ]
