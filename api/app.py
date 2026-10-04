import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="AI Digital Trust Engine")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

@app.on_event("startup")
def preload_models():
    """Load available local models at startup so the first demo click is fast."""
    if SMS_MODEL_PATH.exists():
        try:
            _load_sms()
        except Exception as exc:
            print(f"[Trust Engine] SMS model preload failed: {exc}", flush=True)
    if EMAIL_MODEL_PATH.exists():
        try:
            _load_email()
        except Exception as exc:
            print(f"[Trust Engine] Email model preload failed: {exc}", flush=True)

SMS_MODEL_PATH = Path(os.getenv("DIGITAL_TRUST_SMS_MODEL_PATH", ROOT / "models" / "distilbert_sms_model"))
EMAIL_MODEL_PATH = Path(os.getenv("DIGITAL_TRUST_EMAIL_MODEL_PATH", ROOT / "models" / "distilbert_email_model"))

_sms_model = None
_email_model = None


class AnalyzeRequest(BaseModel):
    type: str
    text: str = ""
    url: str = ""


def _load_sms():
    global _sms_model
    if _sms_model is None:
        print("[Trust Engine] Loading SMS DistilBERT model...", flush=True)
        started = time.time()
        from src.sms_model import SMSModel
        _sms_model = SMSModel(str(SMS_MODEL_PATH))
        print(f"[Trust Engine] SMS model loaded in {time.time()-started:.1f}s", flush=True)
    return _sms_model


def _load_email():
    global _email_model
    if _email_model is None:
        print("[Trust Engine] Loading Email DistilBERT model...", flush=True)
        started = time.time()
        from src.email_model import EmailModel
        _email_model = EmailModel(str(EMAIL_MODEL_PATH))
        print(f"[Trust Engine] Email model loaded in {time.time()-started:.1f}s", flush=True)
    return _email_model


def _warning_signals(text: str) -> list[str]:
    t = (text or "").lower()
    signals = []
    if any(x in t for x in ["urgent", "immediately", "today", "within 24 hours", "expire", "act now"]):
        signals.append("It creates urgency by saying something may happen soon.")
    if any(x in t for x in ["verify", "kyc", "confirm", "update your details", "provide your details"]):
        signals.append("It asks you to verify or provide personal or account information.")
    if any(x in t for x in ["bank", "account", "payment", "card", "kyc"]):
        signals.append("It is asking about a financial or account-related matter.")
    if "http://" in t or "https://" in t or "www." in t:
        signals.append("It contains a link asking you to take action.")
    if any(x in t for x in ["reward", "prize", "won", "congratulations", "cashback"]):
        signals.append("It offers an unexpected reward or benefit.")
    if any(x in t for x in ["blocked", "suspended", "suspension", "restricted"]):
        signals.append("It warns that your account may be blocked, suspended, or restricted.")
    return signals


def _build_result(prediction: str, probabilities: dict[str, float], input_type: str, text: str) -> dict[str, Any]:
    prediction = prediction.lower()
    if input_type == "email":
        fraud_probability = float(probabilities.get("phishing", 0.0))
    else:
        fraud_probability = float(probabilities.get("spam", 0.0)) + float(probabilities.get("smishing", 0.0))

    fraud_score = round(max(0.0, min(1.0, fraud_probability)) * 100, 2)
    risk_level = "HIGH RISK" if fraud_score >= 70 else "MEDIUM RISK" if fraud_score >= 30 else "LOW RISK"

    if input_type == "email":
        unsafe = prediction == "phishing"
    else:
        unsafe = prediction in {"spam", "smishing"}

    if unsafe:
        title = "This message does not look safe."
        summary = f"This {input_type} is highly likely to be fraudulent." if fraud_score >= 70 else f"This {input_type} may be fraudulent."
    else:
        title = "This message looks safe."
        summary = "No significant warning signs were found in this content."

    signals = _warning_signals(text) if unsafe else []
    if unsafe and not signals:
        signals = ["The AI model found patterns associated with fraudulent content."]

    if risk_level == "HIGH RISK":
        actions = [
            "Do not click links or reply to this message.",
            "Do not share your OTP, password, or bank details.",
            "Verify the request through the organization's official app or website.",
        ]
    elif risk_level == "MEDIUM RISK":
        actions = [
            "Do not click links or share personal or financial information yet.",
            "Verify the request through the organization's official app or website.",
            "If unsure, contact the organization using a trusted phone number.",
        ]
    else:
        actions = [
            "You can continue normally, but stay careful with unexpected requests.",
            "Never share your OTP, password, or bank details unexpectedly.",
        ]

    return {
        "title": title,
        "fraud_score": fraud_score,
        "risk_level": risk_level,
        "summary": summary,
        "note": "Score derived from the trained model prediction.",
        "signals": signals,
        "actions": actions,
        "technical": {
            "input_type": input_type,
            "prediction": prediction,
            "probabilities": probabilities,
        },
    }


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "sms_model_available": SMS_MODEL_PATH.exists(),
        "email_model_available": EMAIL_MODEL_PATH.exists(),
    }


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    input_type = request.type.lower()

    if input_type == "sms":
        if not request.text.strip():
            raise HTTPException(400, "Please enter a message.")
        try:
            started = time.time()
            result = _load_sms().predict(request.text)
            print(f"[Trust Engine] SMS inference completed in {time.time()-started:.1f}s", flush=True)
            return _build_result(result["prediction"], result["probabilities"], "sms", request.text)
        except Exception as exc:
            raise HTTPException(500, f"SMS model could not be loaded or run: {exc}")

    if input_type == "email":
        if not request.text.strip():
            raise HTTPException(400, "Please enter an email.")
        try:
            started = time.time()
            result = _load_email().predict(body=request.text)
            print(f"[Trust Engine] Email inference completed in {time.time()-started:.1f}s", flush=True)
            return _build_result(result["prediction"], result["probabilities"], "email", request.text)
        except Exception as exc:
            raise HTTPException(500, f"Email model could not be loaded or run: {exc}")

    if input_type == "url":
        raise HTTPException(501, "The URL model is not connected yet.")

    raise HTTPException(400, "Unsupported input type.")


@app.post("/api/analyze/screenshot")
async def analyze_screenshot(image: UploadFile = File(...)):
    try:
        import io
        import pytesseract
        from PIL import Image

        data = await image.read()
        extracted_text = pytesseract.image_to_string(Image.open(io.BytesIO(data))).strip()
        if not extracted_text:
            raise HTTPException(400, "No readable text was found in the screenshot.")

        result = _load_sms().predict(extracted_text)
        response = _build_result(result["prediction"], result["probabilities"], "screenshot", extracted_text)
        response["technical"]["ocr_text"] = extracted_text
        response["note"] = "Screenshot text was extracted with OCR and analyzed by the trained SMS model."
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Screenshot analysis could not run: {exc}")
