"""
Feature engineering module for creating derived features based on EDA insights.

This module creates business-relevant features that capture:
- Contract risk levels
- Tenure groups
- Payment method risk
- Support service counts
- Charge ratios
- Interaction features

All features are designed to improve model performance and interpretability.
"""

import pandas as pd
import numpy as np


def create_contract_risk_score(df):
    """
    Create contract risk score based on contract type.
    
    Risk levels:
    - Month-to-month = 3 (highest risk)
    - One year = 2 (medium risk)
    - Two year = 1 (lowest risk)
    
    Args:
        df: DataFrame with 'Contract' column
    
    Returns:
        Series: Contract risk scores
    """
    risk_map = {
        'Month-to-month': 3,
        'One year': 2,
        'Two year': 1
    }
    return df['Contract'].map(risk_map).fillna(2)  # Default to medium risk


def create_tenure_groups(df):
    """
    Create tenure groups based on customer lifecycle stage.
    
    Groups:
    - New: 0-12 months (highest churn risk)
    - Short: 13-24 months (medium-high risk)
    - Medium: 25-48 months (medium risk)
    - Long: 49+ months (lowest risk)
    
    Args:
        df: DataFrame with 'tenure' column
    
    Returns:
        Series: Tenure group labels
    """
    return pd.cut(
        df['tenure'],
        bins=[-1, 12, 24, 48, 1000],
        labels=['New', 'Short', 'Medium', 'Long']
    )


def create_payment_risk_score(df):
    """
    Create payment method risk score.
    
    Risk levels:
    - Electronic check = 3 (highest risk)
    - Mailed check = 2 (medium risk)
    - Automatic (Bank transfer, Credit card) = 1 (lowest risk)
    
    Args:
        df: DataFrame with 'PaymentMethod' column
    
    Returns:
        Series: Payment risk scores
    """
    risk_map = {
        'Electronic check': 3,
        'Mailed check': 2,
        'Bank transfer (automatic)': 1,
        'Credit card (automatic)': 1
    }
    return df['PaymentMethod'].map(risk_map).fillna(2)  # Default to medium risk


def create_support_services_count(df):
    """
    Count number of support services a customer has.
    
    Services counted:
    - OnlineSecurity
    - OnlineBackup
    - DeviceProtection
    - TechSupport
    
    Args:
        df: DataFrame with support service columns
    
    Returns:
        Series: Count of support services (0-4)
    """
    support_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    return df[support_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)


def create_charge_per_month(df):
    """
    Create charge per month feature to reduce multicollinearity.
    
    Calculated as: TotalCharges / (tenure + 1)
    This represents average monthly charge over customer lifetime.
    
    Args:
        df: DataFrame with 'TotalCharges' and 'tenure' columns
    
    Returns:
        Series: Charge per month values
    """
    # Avoid division by zero
    return df['TotalCharges'] / (df['tenure'] + 1)


def create_senior_high_charge(df):
    """
    Create binary feature for senior citizens with high charges.
    
    High charge = MonthlyCharges > median
    
    Args:
        df: DataFrame with 'SeniorCitizen' and 'MonthlyCharges' columns
    
    Returns:
        Series: Binary feature (1 if senior with high charges, 0 otherwise)
    """
    median_charges = df['MonthlyCharges'].median()
    return ((df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > median_charges)).astype(int)


def create_fiber_optic_no_support(df):
    """
    Create binary feature for fiber optic customers without support services.
    
    This captures a high-risk combination identified in EDA.
    
    Args:
        df: DataFrame with 'InternetService' and support service columns
    
    Returns:
        Series: Binary feature (1 if fiber optic with no support, 0 otherwise)
    """
    support_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    has_support = df[support_cols].apply(lambda x: (x == 'Yes').any(), axis=1)
    return ((df['InternetService'] == 'Fiber optic') & (~has_support)).astype(int)


def create_all_features(df):
    """
    Create all engineered features and add to dataframe.
    
    Args:
        df: Original dataframe
    
    Returns:
        DataFrame: Dataframe with all engineered features added
    """
    df_eng = df.copy()
    
    # Ensure TotalCharges is numeric
    if df_eng['TotalCharges'].dtype == 'object':
        df_eng['TotalCharges'] = df_eng['TotalCharges'].replace(' ', np.nan)
        df_eng['TotalCharges'] = pd.to_numeric(df_eng['TotalCharges'], errors='coerce')
        df_eng['TotalCharges'] = df_eng['TotalCharges'].fillna(0)
    
    # Create features
    df_eng['contract_risk_score'] = create_contract_risk_score(df_eng)
    df_eng['tenure_group'] = create_tenure_groups(df_eng)
    df_eng['payment_risk_score'] = create_payment_risk_score(df_eng)
    df_eng['support_services_count'] = create_support_services_count(df_eng)
    df_eng['charge_per_month'] = create_charge_per_month(df_eng)
    df_eng['senior_high_charge'] = create_senior_high_charge(df_eng)
    df_eng['fiber_optic_no_support'] = create_fiber_optic_no_support(df_eng)
    
    return df_eng


if __name__ == "__main__":
    # Test feature engineering
    from src.data_processing import load_data, clean_data
    
    print("Testing feature engineering...")
    df = load_data()
    df_clean = clean_data(df)
    df_eng = create_all_features(df_clean)
    
    print(f"\nOriginal columns: {len(df_clean.columns)}")
    print(f"With engineered features: {len(df_eng.columns)}")
    print(f"\nNew features created:")
    new_features = [col for col in df_eng.columns if col not in df_clean.columns]
    for feat in new_features:
        print(f"  - {feat}")
        print(f"    Unique values: {df_eng[feat].nunique()}")
        print(f"    Sample: {df_eng[feat].head(5).tolist()}")
    
    print("\nFeature engineering test complete!")

