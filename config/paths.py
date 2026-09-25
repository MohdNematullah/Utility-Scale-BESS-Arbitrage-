from pathlib import Path

PROJECT__ROOT = Path(__file__).resolve().parents[1]

DATA__DIR = PROJECT__ROOT / "data"
RESULTS__DIR = PROJECT__ROOT / "results"
FIGURE__DIR = PROJECT__ROOT / "figures"
LOG__DIR = PROJECT__ROOT / "logs"
MODEL__DIR = PROJECT__ROOT / "models"

for folder in [RESULTS__DIR, FIGURE__DIR, LOG__DIR, MODEL__DIR]:
    folder.mkdir(parents=True, exist__ok=True)