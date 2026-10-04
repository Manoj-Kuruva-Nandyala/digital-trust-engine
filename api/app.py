from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI(title="AI Digital Trust Engine")

app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")

class AnalyzeRequest(BaseModel):
    type: str
    text: str = ""
    url: str = ""

@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Digital Trust Engine UI is running"}

def demo_result(input_type: str):
    return {
        "title": "This message does not look safe.",
        "fraud_score": 99.38,
        "risk_level": "HIGH RISK",
        "summary": "This message is highly likely to be fraudulent.",
        "note": "Demo UI result. The existing trained models will be connected in the next step.",
        "signals": [
            "It creates urgency by saying something may happen soon.",
            "It asks you to verify or provide account information.",
            "It is asking about your bank or financial account.",
            "It contains a link asking you to take action."
        ],
        "actions": [
            "Do not click the link or reply to this message.",
            "Do not share your OTP, password, or bank details.",
            "Check with your bank using its official app or website."
        ],
        "technical": {"input_type": input_type, "status": "UI prototype"}
    }

@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    # UI-first prototype. Existing SMS/Email models will be wired here next.
    return demo_result(request.type)

@app.post("/api/analyze/screenshot")
async def analyze_screenshot(image: UploadFile = File(...)):
    # OCR + model pipeline will be wired here next.
    return demo_result("screenshot")
