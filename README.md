# ChurnGuard AI v2

Professional churn analytics dashboard — predict, explain, and retain customers.

## What's New in v2

| Feature | Description |
|---|---|
| Single Customer Predictor | Enter any new customer's details manually → instant churn prediction + SHAP + retention plan |
| Predict New CSV | Upload unlabeled active customer data → batch churn predictions with charts + CSV export |
| Model saved as .pkl | After training, model is saved to `saved_model.pkl` automatically |

## Project Structure

```
churnguard_v2/
├── app.py                   ← Main Streamlit app (7 tabs)
├── requirements.txt
├── saved_model.pkl          ← Auto-created after training
├── data/
│   └── generate_sample.py
├── models/
│   └── train.py
└── utils/
    ├── preprocessing.py     ← Now includes preprocess_new_data()
    ├── explainability.py
    └── retention.py
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## The Real Workflow (How to Use It Like a Business)

```
STEP 1: Upload historical data (with Churn column) → Train Model
         ↓
STEP 2a: Tab "Single Customer" → Enter a NEW customer's details → Predict
STEP 2b: Tab "Predict New CSV" → Upload today's active customers → Batch predict
```

This is exactly how real companies use ML churn models.

## Dashboard Tabs

| Tab | Purpose |
|---|---|
| Overview | KPIs, risk distribution, churn histogram |
| Model Performance | ROC curves, confusion matrix, classification report |
| Predictions | Historical predictions with filters + download |
| Explainability | SHAP global importance + per-customer breakdown |
| Retention Strategies | Personalized retention plans + bulk report |
| Single Customer | **NEW** — predict any individual customer manually |
| Predict New CSV | **NEW** — batch predict unlabeled active customer data |
