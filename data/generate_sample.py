import pandas as pd
import numpy as np

def generate_telco_churn(n=1000, seed=42):
    np.random.seed(seed)
    df = pd.DataFrame({
        'customerID': [f'CUST-{i:04d}' for i in range(n)],
        'tenure': np.random.randint(1, 72, n),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, n), 2),
        'TotalCharges': np.round(np.random.uniform(20, 8000, n), 2),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.35, 0.45, 0.20]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'PaymentMethod': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
    })
    churn_prob = (
        0.05
        + 0.25 * (df['Contract'] == 'Month-to-month').astype(int)
        + 0.10 * (df['InternetService'] == 'Fiber optic').astype(int)
        + 0.08 * (df['TechSupport'] == 'No').astype(int)
        + 0.07 * (df['OnlineSecurity'] == 'No').astype(int)
        + 0.05 * (df['PaperlessBilling'] == 'Yes').astype(int)
        + 0.06 * (df['PaymentMethod'] == 'Electronic check').astype(int)
        - 0.15 * (df['tenure'] > 24).astype(int)
        + 0.03 * df['SeniorCitizen']
    ).clip(0.02, 0.95)
    df['Churn'] = (np.random.random(n) < churn_prob).astype(int)
    df['Churn_Label'] = df['Churn'].map({1: 'Yes', 0: 'No'})
    return df

if __name__ == '__main__':
    df = generate_telco_churn(1000)
    df.to_csv('telco_sample.csv', index=False)
    print(f"Generated {len(df)} rows. Churn rate: {df['Churn'].mean():.1%}")
