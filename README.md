# Real-Time Residential Property Price Prediction Pipeline

**DTSC 3601 | Data Science Studio 3: Predictive Analysis & Their Impacts**

**Team members:** Sean McDevitt; Bryson Long; Gauri Shelly Kollassery 

## Overview

This Python project estimates residential property sale prices using housing attributes and a Random Forest regressor. The broader team project uses the RentCast API for property data, Amazon S3 for storage, and SHAP for explaining model predictions. This repository also includes a small, offline training example so anyone can test the prediction workflow without an API key or AWS account.

**Inputs:** bedrooms, bathrooms, square footage, year built, and lot size.  
**Output:** estimated property sale price in US dollars.

> **Important:** The bundled sample dataset is synthetic and is only intended to demonstrate the code. The demo's error metric is not evidence of real-world accuracy, and the predictions are not appraisals or financial advice.

## Required software and libraries

- Python 3.10 or later
- `pandas` — read and clean tabular property data
- `scikit-learn` — train/evaluate a Random Forest model
- `joblib` — save and reload the model
- **Optional live pipeline:** `requests` (RentCast API), `boto3` (AWS S3), `python-dotenv` (local configuration), and `shap` (model explanations)

## Installation

1. Clone your **public** GitHub repository and navigate into it:

   ```bash
   git clone https://github.com/SeanM1212/Charlotte-House-Price-Prediction.git
   cd YOUR-REPOSITORY
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # macOS / Linux:
   source .venv/bin/activate
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```

3. Install the libraries:

   ```bash
   python -m pip install -r requirements.txt
   # Only if you are using live API/S3/SHAP integrations:
   python -m pip install -r requirements-live.txt
   ```

## Get started: train a small demo

Run the offline example using the included synthetic dataset:

```bash
python bin/quickstart.py
```

The script loads `data/sample_listings.csv`, creates a train/test split, trains a Random Forest, prints a sample holdout mean absolute error (MAE), predicts the price for one example property, and saves a model to `artifacts/valuation_model.joblib`. This works without RentCast or AWS credentials.

**Example:** For a property with 3 bedrooms, 2 bathrooms, 1,650 square feet, year built 2005, and a 6,000-square-foot lot, the script prints a predicted dollar amount. The exact amount may vary with data and library versions.

## Live data workflow (team project)

For the RentCast/AWS version, create a local `.env` using `.env.example` as a guide. Provide a RentCast API key and the AWS S3 bucket configuration used by your group. Never commit `.env`, cloud credentials, or API keys.

The live workflow is:

1. Request recent residential listings through the RentCast API.
2. Write the raw response to Amazon S3 and read it back.
3. Extract the five property features and available sale prices.
4. Train or load the Random Forest regression model.
5. Predict values for selected listings and optionally use SHAP to inspect feature contributions.



## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-live.txt
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   └── valuation.py
├── bin/
│   └── quickstart.py
├── data/
│   └── sample_listings.csv
├── docs/
│   └── FUNCTIONS.md
└── tests/
    └── test_valuation.py
```



## Major functions and documentation

- `prepare_features(df)` — select and convert supported housing features to numeric values.
- `train_model(df, random_state=42)` — fit the Random Forest and calculate an illustrative holdout MAE.
- `predict_listing(model, listing)` — generate a price estimate for a supplied listing.
- `save_model(model, path)` / `load_model(path)` — persist and reload the trained model.

Read [function documentation](docs/FUNCTIONS.md) for inputs, outputs, examples, and limitations. Example executable code is in [`bin/quickstart.py`](bin/quickstart.py).

## Responsible use and limitations

Property listings may be incomplete or unrepresentative, especially across neighborhoods or price ranges. Historical market conditions and biased coverage can influence model outputs. Evaluate real-world accuracy on held-out real listings, investigate errors across locations and property types, and use SHAP explanations as supporting diagnostics rather than proof of fairness. The sample training data is **not** suitable for deployment.

## Questions and support

Contact **Sean McDevitt** through DTSC 3602 course/group communications, or open an issue in the team's public GitHub repository. 

## License

See the [LICENSE](LICENSE) file. The included MIT text is a **draft**; 
