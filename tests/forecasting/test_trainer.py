
from data.loader import MarketDataLoader
from features.engineering import FeatureEngineer
from forecasting.trainer import ForecastTrainer


def prepare_data():

    loader = MarketDataLoader()
    engineer = FeatureEngineer()

    df = loader.load_csv(
        "data/raw/ercot_prices.csv"
    )

    return engineer.transform(df)


def test_chronological_split():

    trainer = ForecastTrainer()

    df = prepare_data()

    split = trainer.chronological_split(df)

    assert len(split.train) > 0
    assert len(split.validation) > 0
    assert len(split.test) > 0

    assert (
        split.train.index.max()
        < split.validation.index.min()
    )

    assert (
        split.validation.index.max()
        < split.test.index.min()
    )


def test_walk_forward_generator():

    trainer = ForecastTrainer()

    df = prepare_data()

    folds = list(
        trainer.walk_forward_generator(df)
    )

    assert len(folds) > 0

    train_df, test_df = folds[0]

    assert len(test_df) == 24
    assert len(train_df) > len(test_df)


def test_xgboost_training():

    trainer = ForecastTrainer()

    df = prepare_data()

    split = trainer.chronological_split(df)

    model = trainer.train(
        split.train,
        model_type="xgboost",
    )

    prediction = model.predict(
        split.test.head(5)
    )

    assert len(prediction.prediction) == 5


def test_random_forest_training():

    trainer = ForecastTrainer()

    df = prepare_data()

    split = trainer.chronological_split(df)

    model = trainer.train(
        split.train,
        model_type="random_forest",
    )

    prediction = model.predict(
        split.test.head(5)
    )

    assert len(prediction.prediction) == 5