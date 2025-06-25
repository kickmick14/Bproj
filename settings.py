import os, secrets
from pathlib import Path

# 1) Base directory (this file’s parent directory)
BASE_DIR = Path(__file__).parent.resolve()
# Expose as env var
os.environ.setdefault("BASE_DIR", str(BASE_DIR))

# 2) Model name: can override via environment
MODEL_NAME = os.environ.get("MODEL_NAME", secrets.token_hex(nbytes=8))
os.environ.setdefault("MODEL_NAME", MODEL_NAME)

# 3) Artifacts directory (models)
ARTIFACTS_DIR = Path(
    os.environ.get("ARTIFACTS_DIR", BASE_DIR / "artifacts" / "models")
).resolve()
os.environ.setdefault("ARTIFACTS_DIR", str(ARTIFACTS_DIR))

# 4) Data directory (per-model)
DATA_DIR = Path(
    os.environ.get("DATA_DIR", BASE_DIR / "data" / MODEL_NAME)
).resolve()
os.environ.setdefault("DATA_DIR", str(DATA_DIR))

# 5) Credentials / settings path
CREDENTIALS_PATH = Path(
    os.environ.get("CREDENTIALS_PATH", BASE_DIR / "config" / "settings.json")
).resolve()
os.environ.setdefault("CREDENTIALS_PATH", str(CREDENTIALS_PATH))

# 6) Ensure directories exist
for d in (ARTIFACTS_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)
