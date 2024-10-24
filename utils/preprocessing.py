# utils/preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

def clean_data(calls_df, puts_df):
    # Drop rows with missing data
    calls_clean = calls_df.dropna()
    puts_clean = puts_df.dropna()
    return calls_clean, puts_clean

def feature_engineering(calls_clean):
    # Add features such as time to maturity and risk-free rate
    calls_clean['today'] = datetime.today()
    print(calls_clean.columns)
    calls_clean['expiration'] = pd.to_datetime(calls_clean['expiration'])
    calls_clean['time_to_maturity'] = (calls_clean['expiration'] - calls_clean['today']).dt.days / 365
    calls_clean['risk_free_rate'] = 0.02

    features = calls_clean[['strike', 'lastPrice', 'impliedVolatility', 'time_to_maturity', 'risk_free_rate']]
    return features

def normalize_data(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return pd.DataFrame(scaled_features, columns=features.columns)
