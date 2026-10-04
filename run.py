"""Local launcher for the AI Digital Trust Engine web application.

Run from the repository root:
    python run.py

Then open:
    http://127.0.0.1:8000
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=False)
