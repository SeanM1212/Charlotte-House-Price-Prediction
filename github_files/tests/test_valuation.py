"""Run from repository root: python -m unittest discover -s tests"""
import unittest
from pathlib import Path
import pandas as pd
from src.valuation import prepare_features, train_model, predict_listing


class TestValuation(unittest.TestCase):
    def test_features_order(self):
        df = pd.DataFrame([{ "lotSize": 6000, "yearBuilt": 2005,
            "squareFootage": 1650, "bathrooms": 2, "bedrooms": 3}])
        self.assertEqual(prepare_features(df).columns.tolist(),
                         ["bedrooms", "bathrooms", "squareFootage", "yearBuilt", "lotSize"])

    def test_train_and_predict(self):
        df = pd.read_csv(Path(__file__).resolve().parents[1]/"data"/"sample_listings.csv")
        model, mae = train_model(df)
        self.assertGreaterEqual(mae, 0)
        self.assertGreater(predict_listing(model, df.iloc[0].to_dict()), 0)


if __name__ == "__main__":
    unittest.main()
