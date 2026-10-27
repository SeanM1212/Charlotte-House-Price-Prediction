"""Reproducible educational Random Forest property-price demo.

This offline module does not call RentCast, AWS, or SHAP. It is deliberately
separate from any group-specific cloud pipeline and needs only local sample data.
"""
from pathlib import Path
from typing import Mapping, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

FEATURE_COLUMNS = ["bedrooms", "bathrooms", "squareFootage", "yearBuilt", "lotSize"]
TARGET_COLUMN = "price"


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric property features in the model's required column order.

    Missing columns raise ValueError; missing feature *values* are imputed by
    the fitted Pipeline instead of being dropped or filled using test data.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")


def train_model(df: pd.DataFrame, random_state: int = 42):
    """Train a Random Forest from a DataFrame and return (model, holdout_MAE).

    Models are fitted only on the training split; errors are calculated against
    unseen demo rows. Small synthetic samples give unstable, illustrative MAE.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset is missing target column: {TARGET_COLUMN}")
    x = prepare_features(df)
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    valid = y.notna()
    x, y = x.loc[valid], y.loc[valid]
    if len(y) < 10:
        raise ValueError("At least 10 rows with valid prices are needed for this demo")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=random_state
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=random_state)),
    ])
    model.fit(x_train, y_train)
    mae = float(mean_absolute_error(y_test, model.predict(x_test)))
    return model, mae


def predict_listing(model: Pipeline, listing: Union[Mapping, pd.DataFrame]) -> float:
    """Predict one listing's price in USD using the same feature ordering."""
    df = listing if isinstance(listing, pd.DataFrame) else pd.DataFrame([listing])
    if len(df) != 1:
        raise ValueError("predict_listing expects exactly one listing")
    return float(model.predict(prepare_features(df))[0])


def save_model(model: Pipeline, path: Union[str, Path]) -> None:
    """Save a model locally; load only trusted model files."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: Union[str, Path]) -> Pipeline:
    """Load a previously saved trusted model file."""
    return joblib.load(path)
