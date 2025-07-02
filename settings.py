#######################################
# @author Michael Kane
# @date 27/06/2025
# Sets environmental variables
#######################################
import os, uuid
from pathlib import Path

# 1) Base directory (this file’s parent directory)
BASE_DIR = Path(__file__).parent.resolve()
os.environ.setdefault("BASE_DIR", str(BASE_DIR))

# 2) Model name: can override via environment
MODEL_NAME = os.environ.get("MODEL_NAME", "FAILSAFE_MODEL_NAME")
os.environ.setdefault("MODEL_NAME", MODEL_NAME)

# 3) Artifacts directory (models)
ARTIFACTS_DIR = Path(
    os.environ.get("ARTIFACTS_DIR", BASE_DIR / "artifacts" / "models")
).resolve()
os.environ.setdefault("ARTIFACTS_DIR", str(ARTIFACTS_DIR))
os.makedirs(ARTIFACTS_DIR, exist_ok=True)   

# 4) Data directory (per-model)
DATA_DIR = Path(
    os.environ.get("DATA_DIR", BASE_DIR / "data" / MODEL_NAME)
).resolve()
os.environ.setdefault("DATA_DIR", str(DATA_DIR))
os.makedirs(f"{DATA_DIR}", exist_ok=True)

# 5) Credentials / settings path
CREDENTIALS_PATH = Path(
    os.environ.get("CREDENTIALS_PATH", BASE_DIR / "config" / "settings.json")
).resolve()
os.environ.setdefault("CREDENTIALS_PATH", str(CREDENTIALS_PATH))
