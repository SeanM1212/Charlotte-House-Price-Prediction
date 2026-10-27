# Function documentation

This file describes the runnable **offline reference module** in `src/valuation.py`.
The group may also have separate live ingestion files (`demo.py`, `model_utils.py`,
`train_model.py`); document their actual functions and signatures after uploading
those original files to the public repository.

## `prepare_features(df)`

- **Input:** `pandas.DataFrame` with columns `bedrooms`, `bathrooms`, `squareFootage`, `yearBuilt`, and `lotSize`.
- **Output:** Numeric DataFrame containing those five columns, in that order.
- **Behavior:** Converts strings to numeric where possible; unparseable values become missing values. Raises `ValueError` if a required column is absent.
- **Reason:** The same ordered features must be used for fitting and inference. A model pipeline handles missing values.

## `train_model(df, random_state=42)`

- **Input:** Property DataFrame with five feature columns plus the `price` target; at least 10 valid target rows.
- **Output:** A tuple `(fitted_model, holdout_mae)`.
- **Behavior:** Removes rows with unavailable prices; performs a reproducible 75/25 train/test split; fits median imputation and `RandomForestRegressor(n_estimators=100)` to the training split; computes holdout MAE in dollars.
- **Limitation:** Error on a small synthetic dataset is educational and not representative of Charlotte market accuracy.

## `predict_listing(model, listing)`

- **Input:** Trained pipeline and a dictionary or one-row DataFrame with the five features.
- **Output:** One estimated price as a Python float, in USD.
- **Behavior:** Ensures feature ordering matches training, applies the fitted imputer, and predicts with the fitted regressor.

## `save_model(model, path)` and `load_model(path)`

- **Input:** A fitted model and an output path for saving; a trusted joblib file path for loading.
- **Output:** `save_model` writes a file; `load_model` returns the stored model.
- **Security:** Do not load joblib files from unknown or untrusted sources.

## Example usage

```python
import pandas as pd
from src.valuation import train_model, predict_listing

df = pd.read_csv("data/sample_listings.csv")
model, mae = train_model(df)
new_house = {
    "bedrooms": 3, "bathrooms": 2, "squareFootage": 1650,
    "yearBuilt": 2005, "lotSize": 6000,
}
print("MAE:", mae)
print("Prediction:", predict_listing(model, new_house))
```

For a complete executable script, see [`bin/quickstart.py`](../bin/quickstart.py).

## Broader RentCast/AWS architecture

The **group's live pipeline** uses RentCast REST requests, S3 storage, numeric
feature preparation, Random Forest fitting, and optional SHAP explanations.
Its implementation and parameter names should be documented from the group's
actual source, not assumed to be identical to the offline example above.

Possible evaluation safeguards: time-aware data splits, checks for duplicate
listings, comparison of neighborhood-level error, and inspection of missing
features and outliers. SHAP explains model outputs but does not establish fairness.
