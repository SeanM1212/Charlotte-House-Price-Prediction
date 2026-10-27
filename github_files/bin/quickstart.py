"""Offline example: load sample listings, train/evaluate, predict, and save.

Run from the repository root with: python bin/quickstart.py
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.valuation import train_model, predict_listing, save_model, load_model


def main() -> None:
    data_path = ROOT / "data" / "sample_listings.csv"
    listings = pd.read_csv(data_path)
    model, mae = train_model(listings)
    print(f"Loaded {len(listings)} synthetic sample listings")
    print(f"Illustrative holdout MAE: ${mae:,.2f}")

    example = {
        "bedrooms": 3, "bathrooms": 2, "squareFootage": 1650,
        "yearBuilt": 2005, "lotSize": 6000,
    }
    predicted = predict_listing(model, example)
    print(f"Example predicted property price: ${predicted:,.2f}")

    output = ROOT / "artifacts" / "valuation_model.joblib"
    save_model(model, output)
    loaded = load_model(output)
    assert abs(predict_listing(loaded, example) - predicted) < 1e-8
    print(f"Saved trained model to {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
