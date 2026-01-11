"""
Preprocessing pipelines for feature engineering and transformation.

**Preprocessing Rationale:**

1. **Numerical Feature Scaling (StandardScaler):**
   - **Why:** Logistic Regression uses gradient descent and is sensitive to feature scales
   - **When:** Applied only for Logistic Regression (scale_numerical=True)
   - **Why not for tree models:** Random Forest and XGBoost are scale-invariant
   - **Method:** StandardScaler (z-score normalization: mean=0, std=1)
   - **Rationale:** Preserves distribution shape, handles outliers better than MinMaxScaler

2. **Categorical Feature Encoding (OneHotEncoder):**
   - **Why:** Machine learning models require numerical input
   - **Method:** OneHotEncoder with drop='first'
   - **Why drop='first':** Avoids dummy variable trap (multicollinearity)
     - If we have N categories, we create N-1 binary features
     - The dropped category becomes the reference category
   - **handle_unknown='ignore':** Handles new categories in test set gracefully

3. **Separate Pipelines for Different Models:**
   - **Why:** Different models have different preprocessing needs
   - Logistic Regression: Needs scaling → scale_numerical=True
   - Tree models: Don't need scaling → scale_numerical=False
   - **Benefit:** Optimizes preprocessing for each model type

4. **Data Leakage Prevention:**
   - Preprocessor is fitted ONLY on training data
   - Test data is transformed using fitted preprocessor
   - **Critical:** Never fit on test data to avoid data leakage

This module creates sklearn pipelines for:
- Numerical feature scaling (StandardScaler)
- Categorical feature encoding (OneHotEncoder)
- Combined preprocessing using ColumnTransformer
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_numerical_features():
    """Return list of numerical feature names."""
    return ['tenure', 'MonthlyCharges', 'TotalCharges']


def get_categorical_features(df):
    """
    Identify categorical features (all non-numerical, non-target columns).
    
    Args:
        df: Dataframe to analyze
    
    Returns:
        list: Categorical feature names
    """
    numerical = get_numerical_features()
    categorical = [col for col in df.columns if col not in numerical]
    return categorical


def create_preprocessor(scale_numerical=True):
    """
    Create a preprocessing pipeline using ColumnTransformer.
    
    Args:
        scale_numerical: Whether to scale numerical features (needed for Logistic Regression)
    
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Numerical features pipeline
    if scale_numerical:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = 'passthrough'  # No scaling for tree models
    
    # Categorical features pipeline
    categorical_transformer = OneHotEncoder(
        drop='first',  # Avoid multicollinearity
        sparse_output=False,  # Return numpy array instead of sparse matrix
        handle_unknown='ignore'  # Handle unseen categories in test set
    )
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, get_numerical_features()),
            ('cat', categorical_transformer, get_categorical_features)
        ],
        remainder='drop'  # Drop any columns not explicitly handled
    )
    
    return preprocessor


def create_preprocessing_pipeline(model, scale_numerical=True):
    """
    Create a complete pipeline combining preprocessing and model.
    
    Args:
        model: sklearn model instance
        scale_numerical: Whether to scale numerical features
    
    Returns:
        Pipeline: Combined preprocessing and model pipeline
    """
    preprocessor = create_preprocessor(scale_numerical=scale_numerical)
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline


def get_categorical_features_for_preprocessor(X):
    """Helper function to get categorical features (must be at module level for pickling)."""
    numerical = get_numerical_features()
    return [col for col in X.columns if col not in numerical]


def fit_preprocessor(X_train, scale_numerical=True):
    """
    Fit the preprocessor on training data.
    
    Args:
        X_train: Training features dataframe
        scale_numerical: Whether to scale numerical features
    
    Returns:
        ColumnTransformer: Fitted preprocessor
    """
    # Create preprocessor with proper categorical feature list
    if scale_numerical:
        numerical_transformer = StandardScaler()
    else:
        numerical_transformer = 'passthrough'
    
    categorical_transformer = OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    )
    
    # Get categorical features list (not function, for pickling compatibility)
    cat_features = get_categorical_features_for_preprocessor(X_train)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, get_numerical_features()),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='drop'
    )
    
    # Fit on training data
    preprocessor.fit(X_train)
    
    return preprocessor


def transform_data(preprocessor, X):
    """
    Transform data using fitted preprocessor.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        X: Dataframe to transform
    
    Returns:
        np.ndarray: Transformed features
    """
    return preprocessor.transform(X)


def get_feature_names(preprocessor, X_train):
    """
    Get feature names after preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        X_train: Training dataframe (to get original column names)
    
    Returns:
        list: Feature names after encoding
    """
    feature_names = []
    
    # Numerical features
    num_features = get_numerical_features()
    feature_names.extend(num_features)
    
    # Categorical features - need to get them from the preprocessor
    # The preprocessor was fitted with a callable, so we need to call it
    if hasattr(preprocessor.transformers_[1][2], '__call__'):
        # It's a callable, call it with X_train
        cat_features = preprocessor.transformers_[1][2](X_train)
    else:
        # It's a list
        cat_features = preprocessor.transformers_[1][2]
    
    cat_transformer = preprocessor.named_transformers_['cat']
    
    if hasattr(cat_transformer, 'get_feature_names_out'):
        # sklearn >= 1.0
        cat_feature_names = cat_transformer.get_feature_names_out(cat_features)
        feature_names.extend(cat_feature_names)
    else:
        # Fallback for older sklearn versions
        for i, feature in enumerate(cat_features):
            categories = cat_transformer.categories_[i]
            # Skip first category (dropped)
            for cat in categories[1:]:
                feature_names.append(f"{feature}_{cat}")
    
    return feature_names


if __name__ == "__main__":
    # Test preprocessing
    from src.data_processing import load_and_process_data
    
    print("Testing preprocessing pipeline...")
    X_train, X_test, y_train, y_test = load_and_process_data(save_splits_flag=False)
    
    # Test with scaling (for Logistic Regression)
    print("\n1. Testing preprocessor with scaling:")
    preprocessor_scaled = fit_preprocessor(X_train, scale_numerical=True)
    X_train_scaled = transform_data(preprocessor_scaled, X_train)
    X_test_scaled = transform_data(preprocessor_scaled, X_test)
    print(f"   Scaled train shape: {X_train_scaled.shape}")
    print(f"   Scaled test shape: {X_test_scaled.shape}")
    
    # Test without scaling (for tree models)
    print("\n2. Testing preprocessor without scaling:")
    preprocessor_unscaled = fit_preprocessor(X_train, scale_numerical=False)
    X_train_unscaled = transform_data(preprocessor_unscaled, X_train)
    X_test_unscaled = transform_data(preprocessor_unscaled, X_test)
    print(f"   Unscaled train shape: {X_train_unscaled.shape}")
    print(f"   Unscaled test shape: {X_test_unscaled.shape}")
    
    print("\nPreprocessing test complete!")

