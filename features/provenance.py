
"""
provenance.py
=============

Feature registry for .

Stores metadata describing every engineered feature
used by forecasting models.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class FeatureInfo:
    name: str
    category: str
    description: str
    unit: str


FEATURE_REGISTRY = [

    # =====================================================
    # Calendar Features
    # =====================================================

    FeatureInfo(
        "hour",
        "calendar",
        "Hour of day (0â€“23).",
        "hour",
    ),
    FeatureInfo(
        "day_of_week",
        "calendar",
        "Day of week (0=Monday).",
        "index",
    ),
    FeatureInfo(
        "day_of_month",
        "calendar",
        "Day of month.",
        "day",
    ),
    FeatureInfo(
        "day_of_year",
        "calendar",
        "Day within year.",
        "day",
    ),
    FeatureInfo(
        "week_of_year",
        "calendar",
        "ISO week number.",
        "week",
    ),
    FeatureInfo(
        "month",
        "calendar",
        "Month number.",
        "month",
    ),
    FeatureInfo(
        "quarter",
        "calendar",
        "Calendar quarter.",
        "quarter",
    ),
    FeatureInfo(
        "is_weekend",
        "calendar",
        "Weekend indicator.",
        "binary",
    ),

    # =====================================================
    # Cyclical Features
    # =====================================================

    FeatureInfo(
        "hour_sin",
        "cyclical",
        "Sine encoding of hour.",
        "scaled",
    ),
    FeatureInfo(
        "hour_cos",
        "cyclical",
        "Cosine encoding of hour.",
        "scaled",
    ),
    FeatureInfo(
        "weekday_sin",
        "cyclical",
        "Sine encoding of weekday.",
        "scaled",
    ),
    FeatureInfo(
        "weekday_cos",
        "cyclical",
        "Cosine encoding of weekday.",
        "scaled",
    ),
    FeatureInfo(
        "month_sin",
        "cyclical",
        "Sine encoding of month.",
        "scaled",
    ),
    FeatureInfo(
        "month_cos",
        "cyclical",
        "Cosine encoding of month.",
        "scaled",
    ),

    # =====================================================
    # Lag Features
    # =====================================================

    *[
        FeatureInfo(
            f"price_lag_{lag}",
            "lag",
            f"Electricity price {lag} hour(s) ago.",
            "USD/MWh",
        )
        for lag in (1, 2, 3, 6, 12, 24, 48, 72, 168)
    ],

    # =====================================================
    # Rolling Statistics
    # =====================================================

    *[
        FeatureInfo(
            f"price_mean_{window}",
            "rolling_mean",
            f"Mean electricity price over previous {window} hours.",
            "USD/MWh",
        )
        for window in (3, 6, 12, 24, 48, 168)
    ],

    *[
        FeatureInfo(
            f"price_std_{window}",
            "rolling_std",
            f"Price volatility over previous {window} hours.",
            "USD/MWh",
        )
        for window in (3, 6, 12, 24, 48, 168)
    ],

    *[
        FeatureInfo(
            f"price_min_{window}",
            "rolling_min",
            f"Minimum price over previous {window} hours.",
            "USD/MWh",
        )
        for window in (3, 6, 12, 24, 48, 168)
    ],

    *[
        FeatureInfo(
            f"price_max_{window}",
            "rolling_max",
            f"Maximum price over previous {window} hours.",
            "USD/MWh",
        )
        for window in (3, 6, 12, 24, 48, 168)
    ],

    # =====================================================
    # Price Dynamics
    # =====================================================

    FeatureInfo(
        "price_change_1h",
        "price_dynamics",
        "Previous hourly price difference.",
        "USD/MWh",
    ),
    FeatureInfo(
        "price_change_24h",
        "price_dynamics",
        "Difference from previous day.",
        "USD/MWh",
    ),
    FeatureInfo(
        "price_momentum_6h",
        "price_dynamics",
        "Six-hour price momentum.",
        "USD/MWh",
    ),
    FeatureInfo(
        "price_volatility_24h",
        "price_dynamics",
        "24-hour rolling standard deviation.",
        "USD/MWh",
    ),
]


def feature_names():
    return [feature.name for feature in FEATURE_REGISTRY]


def feature_categories():
    categories = {}

    for feature in FEATURE_REGISTRY:
        categories.setdefault(feature.category, []).append(feature.name)

    return categories


def feature_table():
    import pandas as pd

    return pd.DataFrame(
        [
            vars(feature)
            for feature in FEATURE_REGISTRY
        ]
    )