from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "figures"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"

for folder in [RESULTS_DIR, FIGURE_DIR, LOG_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)