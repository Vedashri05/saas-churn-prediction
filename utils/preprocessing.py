import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def auto_detect_churn_column(df):
    """Try to auto-detect the churn/target column."""
    candidates = [c for c in df.columns if any(
        kw in c.lower() for kw in ['churn', 'churned', 'attrition', 'leave', 'exit', 'target', 'label']
    )]
    if candidates:
        return candidates[0]
    return df.columns[-1]

def preprocess(df, target_col, id_col=None, fit_scaler=True, scaler=None, encoders=None):
    df = df.copy()

    drop_cols = []
    if id_col:
        drop_cols.append(id_col)
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == object and df[col].nunique() == len(df):
            drop_cols.append(col)
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

    y_raw = df[target_col].copy()
    df.drop(columns=[target_col], inplace=True)

    if y_raw.dtype == object:
        y_raw = y_raw.str.strip().str.lower()
        y = y_raw.map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        if y.isna().any():
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw), name=target_col)
    else:
        y = y_raw.astype(int)

    feature_names = list(df.columns)

    if encoders is None:
        encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in df.select_dtypes(include='object').columns:
            if col in encoders:
                le = encoders[col]
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(0, inplace=True)

    X = df.values.astype(float)

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaler is not None:
        X = scaler.transform(X)

    return X, y.values, feature_names, scaler, encoders

def preprocess_new_data(df, feature_names, scaler, encoders):
    """
    Preprocess a new unlabeled CSV using the already-fitted scaler and encoders.
    Aligns columns to match training features exactly.
    """
    df = df.copy()

    # Drop ID-like columns
    for col in df.columns:
        if df[col].dtype == object and df[col].nunique() == len(df):
            df.drop(columns=[col], inplace=True, errors='ignore')

    # Encode categoricals using saved encoders
    for col in df.select_dtypes(include='object').columns:
        if col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(df[col])
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df.fillna(df.median(numeric_only=True), inplace=True)
    df.fillna(0, inplace=True)

    # Align columns to training features
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    X = df.values.astype(float)
    X = scaler.transform(X)
    return X

def get_feature_df(df, target_col, id_col=None):
    drop_cols = [target_col]
    if id_col:
        drop_cols.append(id_col)
    for col in df.columns:
        if col not in drop_cols and df[col].dtype == object and df[col].nunique() == len(df):
            drop_cols.append(col)
    return df.drop(columns=drop_cols, errors='ignore')
