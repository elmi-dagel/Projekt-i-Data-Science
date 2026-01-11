"""
Data loading, cleaning, and splitting utilities.

This module handles:
- Loading raw CSV data
- Cleaning missing values and data types
- Creating stratified train/test splits
- Saving processed splits for reproducibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(data_path=None):
    """
    Load the Telco Customer Churn dataset.
    
    Args:
        data_path: Path to the CSV file. If None, uses default location.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if data_path is None:
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "raw" / "telco.csv"
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Please run: python -m src.download_data"
        )
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def clean_data(df):
    """
    Clean the dataset by handling missing values and data types.
    
    **Documented Cleaning Decisions:**
    
    1. **Problem:** customerID is an identifier, not a predictive feature
       **Solution:** Drop customerID column
       **Rationale:** Customer IDs are unique identifiers and would cause overfitting
       if used as a feature. They don't provide predictive value.
    
    2. **Problem:** TotalCharges stored as object type with empty strings (' ')
       **Solution:** Convert to numeric, handle missing values
       **Rationale:** 
       - Empty strings occur for new customers (tenure=0) who haven't been charged yet
       - For tenure=0: Set TotalCharges = 0 (logically correct)
       - For tenure > 0: Calculate from MonthlyCharges × tenure (more accurate)
       - Only use median as fallback if MonthlyCharges is also missing
       - This preserves the relationship: TotalCharges ≈ MonthlyCharges × tenure
    
    3. **Outlier Treatment:** 
       - Decision: Keep outliers (no capping/removal)
       - Rationale: Outliers may represent legitimate high-value customers or edge cases
       - Tree-based models (RF, XGBoost) are robust to outliers
       - If needed, can be addressed in preprocessing for Logistic Regression
    
    Args:
        df: Raw dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\nCleaning data...")
    print("="*60)
    df = df.copy()
    
    # Decision 1: Drop customerID (not a predictive feature)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("✓ Dropped customerID column (identifier, not a feature)")
    
    # Decision 2: Handle TotalCharges data type and missing values
    # Problem: TotalCharges stored as object with empty strings
    if df['TotalCharges'].dtype == 'object':
        print("\n⚠️  Data Quality Issue: TotalCharges is object type")
        print("   Solution: Converting to numeric and handling missing values")
        # Replace empty strings with NaN
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        # Convert to numeric
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges
    missing_count = df['TotalCharges'].isna().sum()
    if missing_count > 0:
        print(f"\nFound {missing_count} missing values in TotalCharges")
        # For new customers (tenure=0), TotalCharges should be 0
        new_customers = (df['tenure'] == 0).sum()
        df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
        print(f"  - Set {new_customers} new customers (tenure=0) to TotalCharges=0")
        
        # For customers with tenure > 0 but missing TotalCharges:
        # Calculate from MonthlyCharges × tenure (more accurate than median)
        mask = (df['TotalCharges'].isna()) & (df['tenure'] > 0)
        calculated_count = mask.sum()
        if calculated_count > 0:
            df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * df.loc[mask, 'tenure']
            print(f"  - Calculated {calculated_count} missing values from MonthlyCharges × tenure")
        
        # Only use median as fallback if MonthlyCharges is also missing
        remaining_missing = df['TotalCharges'].isna().sum()
        if remaining_missing > 0:
            median_value = df['TotalCharges'].median()
            df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
            print(f"  - Filled {remaining_missing} remaining with median ({median_value:.2f}) as fallback")
        print("✓ TotalCharges cleaning complete")
    
    # Ensure tenure and MonthlyCharges are numeric
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    # Decision 3: Outlier treatment
    print("\n📊 Outlier Treatment Decision:")
    print("   Decision: Keep all outliers (no capping/removal)")
    print("   Rationale:")
    print("     - Outliers may represent legitimate high-value customers")
    print("     - Tree-based models (RF, XGBoost) are robust to outliers")
    print("     - Preserves data integrity for business insights")
    
    # Check for any remaining missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️  Warning: Remaining missing values:")
        print(missing[missing > 0])
    else:
        print("\n✓ No missing values remaining")
    
    # Display data info
    print(f"\n{'='*60}")
    print(f"Data shape after cleaning: {df.shape}")
    print(f"Churn distribution:")
    print(df['Churn'].value_counts())
    print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.2%}")
    print(f"{'='*60}")
    
    return df


def split_data(df, test_size=0.2, validation_size=0.2, random_state=42, stratify_col='Churn', use_validation=False):
    """
    Split data into train/validation/test sets with stratification.
    
    Args:
        df: Cleaned dataframe
        test_size: Proportion of data for test set (default 0.2)
        validation_size: Proportion of data for validation set (default 0.2, only if use_validation=True)
        random_state: Random seed for reproducibility
        stratify_col: Column to stratify on (default 'Churn')
        use_validation: Whether to create a validation set (60/20/20 split)
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) if use_validation=True
               (X_train, X_test, y_train, y_test) if use_validation=False
    """
    if use_validation:
        print(f"\nSplitting data (train/val/test: 60/20/20, random_state={random_state})...")
    else:
        print(f"\nSplitting data (train/test: {1-test_size:.0%}/{test_size:.0%}, random_state={random_state})...")
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    if use_validation:
        # First split: train+val (80%) vs test (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: train (75% of temp = 60% of total) vs val (25% of temp = 20% of total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=validation_size / (1 - test_size),  # 0.25 to get 20% of total
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"Train set: {len(X_train)} samples ({len(X_train)/len(df):.1%})")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df):.1%})")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df):.1%})")
        print(f"\nTrain Churn distribution:")
        print(y_train.value_counts())
        print(f"Validation Churn distribution:")
        print(y_val.value_counts())
        print(f"Test Churn distribution:")
        print(y_test.value_counts())
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        # Stratified split to maintain churn ratio
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"Train set: {len(X_train)} samples ({len(X_train)/len(df):.1%})")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df):.1%})")
        print(f"\nTrain Churn distribution:")
        print(y_train.value_counts())
        print(f"Test Churn distribution:")
        print(y_test.value_counts())
        
        return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test, output_dir=None):
    """
    Save train/test splits to CSV files for reproducibility.
    
    Args:
        X_train, X_test: Feature dataframes
        y_train, y_test: Target series
        output_dir: Directory to save splits. If None, uses default location.
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "data" / "processed"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target for saving
    train_df = X_train.copy()
    train_df['Churn'] = y_train
    
    test_df = X_test.copy()
    test_df['Churn'] = y_test
    
    # Save to CSV
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nSaved train set to {train_path}")
    print(f"Saved test set to {test_path}")


def load_and_process_data(save_splits_flag=True, use_feature_engineering=True, use_validation=False):
    """
    Complete data loading and processing pipeline.
    
    Args:
        save_splits_flag: Whether to save train/test splits to disk
        use_feature_engineering: Whether to create engineered features
        use_validation: Whether to create a validation set (60/20/20 split)
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test) if use_validation=True
               (X_train, X_test, y_train, y_test) if use_validation=False
    """
    # Load data
    df = load_data()
    
    # Clean data
    df_clean = clean_data(df)
    
    # Feature engineering (optional)
    if use_feature_engineering:
        from src.feature_engineering import create_all_features
        df_clean = create_all_features(df_clean)
        print("\n✓ Applied feature engineering")
    
    # Split data
    splits = split_data(df_clean, use_validation=use_validation)
    
    if use_validation:
        X_train, X_val, X_test, y_train, y_val, y_test = splits
        # Save splits if requested
        if save_splits_flag:
            save_splits(X_train, X_test, y_train, y_test)  # Still save train/test
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        X_train, X_test, y_train, y_test = splits
        # Save splits if requested
        if save_splits_flag:
            save_splits(X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the data processing pipeline
    X_train, X_test, y_train, y_test = load_and_process_data()
    print("\nData processing complete!")

