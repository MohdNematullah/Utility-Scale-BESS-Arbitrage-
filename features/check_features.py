"""
check_features.py

Feature diagnostics for .

Can be executed in two ways:

1. python -m features.check_features   (recommended)
2. python features/check_features.py   (also supported)
"""

from pathlib import Path
import sys

# -------------------------------------------------------
# Allow running as a standalone script
# -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer


def main():

    loader = MarketDataLoader()

    df = loader.load_csv("data/raw/ercot_prices.csv")

    engineer = FeatureEngineer()

    features = engineer.transform(df)

    print("=" * 70)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 70)

    print(f"Rows                 : {len(features)}")
    print(f"Columns              : {len(features.columns)}")
    print(f"Missing Values       : {features.isna().sum().sum()}")
    print(f"Feature Columns      : {len(engineer.feature_columns(features))}")

    print("\nFirst 15 Features")
    for column in features.columns[:15]:
        print(f"  â€¢ {column}")

    print("\nLast 5 Features")
    for column in features.columns[-5:]:
        print(f"  â€¢ {column}")

    print("\nFeature Matrix Shape")
    print(features.shape)

    output = Path("features/feature_matrix_preview.csv")
    features.head(200).to_csv(output)

    print("\nPreview saved to:")
    print(output)


if __name__ == "__main__":
    main()