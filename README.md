# AI Digital Trust Engine

Case Study: **AI-Based Digital Trust Engine for Protecting Senior Citizens from Phishing and Digital Scams**

The project provides a simple user-facing safety assistant based on the flow:

**DETECT → EXPLAIN → GUIDE**

## Run the demo locally

You do **not** need Google Colab to run the application.

### 1. Clone the repository

```bash
git clone https://github.com/Manoj-Kuruva-Nandyala/digital-trust-engine.git
cd digital-trust-engine
```

### 2. Create a Python environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the web application dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
python run.py
```

Open **http://127.0.0.1:8000** in a browser.

The application is designed so the UI can be presented directly from a local machine without running notebook cells.

## Project structure

- `web/` — user interface
- `api/` — FastAPI application
- `src/` — AI models, risk engine, XAI and guidance
- `training/` — model training scripts
- `models/` — model storage location
- `notebooks/` — experiments and development notebooks
- `run.py` — one-command local launcher

## Model files

Trained model binaries are intentionally not committed to GitHub. They should be stored locally and referenced through environment variables when the model integration is enabled.

See `.env.example` for the expected paths.

Google Colab remains useful for GPU training and experimentation, but it is **not required for the final demo application**.
