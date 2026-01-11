"""
Model training and hyperparameter tuning.

**Model Selection Rationale:**

1. **Logistic Regression (Baseline Model):**
   - **Why:** Interpretable, provides probability estimates, good baseline
   - **Advantages:** 
     - Highly interpretable (coefficients show feature impact)
     - Fast training and prediction
     - Provides calibrated probabilities
     - Good for understanding feature relationships
   - **Disadvantages:**
     - Assumes linear relationships
     - Requires feature scaling
     - May struggle with complex non-linear patterns
   - **Use case:** Baseline model, interpretability-focused scenarios

2. **Random Forest:**
   - **Why:** Handles non-linear relationships, provides feature importance
   - **Advantages:**
     - Captures non-linear patterns and interactions
     - Robust to outliers (no scaling needed)
     - Provides feature importance
     - Handles mixed data types well
     - Less prone to overfitting than single trees
   - **Disadvantages:**
     - Less interpretable than Logistic Regression
     - Can be memory intensive
     - May overfit with many trees
   - **Use case:** Balanced performance and interpretability

3. **XGBoost (Extreme Gradient Boosting):**
   - **Why:** State-of-the-art performance, handles imbalanced data well
   - **Advantages:**
     - Excellent predictive performance
     - Built-in handling of class imbalance (scale_pos_weight)
     - Robust to outliers
     - Feature importance available
     - Regularization prevents overfitting
   - **Disadvantages:**
     - Less interpretable
     - More hyperparameters to tune
     - Longer training time
   - **Use case:** Maximum predictive performance

**Class Imbalance Handling:**
- **Why class_weight='balanced':** Dataset has ~26.5% churn (imbalanced)
- **Effect:** Automatically adjusts class weights inversely proportional to frequency
- **Benefit:** Models focus more on minority class (churners)
- **Alternative:** SMOTE (not used here, but could be considered)

**Hyperparameter Tuning Strategy:**
- **Focus:** Recall and F1-score (business-critical metrics)
- **Why Recall:** We want to catch as many churners as possible (minimize False Negatives)
- **Why F1:** Balances precision and recall
- **Method:** GridSearchCV (Logistic Regression) / RandomizedSearchCV (RF, XGBoost)
- **CV folds:** 5-fold cross-validation for robust evaluation

This module trains multiple models:
- Logistic Regression (baseline, with scaling)
- Random Forest (no scaling needed)
- XGBoost (no scaling needed)

All models are tuned for recall and F1-score on the churn class.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score, f1_score
import joblib
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


def create_scoring_dict():
    """
    Create dictionary of scoring metrics for model evaluation.
    Focus on recall and F1-score for churn class.
    
    Returns:
        dict: Scoring metrics
    """
    return {
        'recall': make_scorer(recall_score, pos_label='Yes'),
        'f1': make_scorer(f1_score, pos_label='Yes'),
        'accuracy': 'accuracy'
    }


def train_logistic_regression(X_train, y_train, cv=5, random_state=42):
    """
    Train Logistic Regression model with hyperparameter tuning.
    
    Args:
        X_train: Training features (already preprocessed/scaled)
        y_train: Training target
        cv: Cross-validation folds
        random_state: Random seed
    
    Returns:
        tuple: (best_model, best_params, cv_results)
    """
    print("\n" + "="*60)
    print("Training Logistic Regression (Baseline Model)")
    print("="*60)
    
    # Logistic Regression with balanced class weights
    base_model = LogisticRegression(
        class_weight='balanced',
        random_state=random_state,
        max_iter=1000
    )
    
    # Hyperparameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs', 'liblinear']
    }
    
    # Use F1-score as primary metric for selection
    scoring = create_scoring_dict()
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit='f1',  # Refit best model using F1-score
        n_jobs=-1,
        verbose=1
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
    
    return best_model, best_params, grid_search.cv_results_


def train_random_forest(X_train, y_train, cv=5, random_state=42, n_iter=20):
    """
    Train Random Forest model with hyperparameter tuning.
    
    Args:
        X_train: Training features (preprocessed, no scaling needed)
        y_train: Training target
        cv: Cross-validation folds
        random_state: Random seed
        n_iter: Number of iterations for randomized search
    
    Returns:
        tuple: (best_model, best_params, cv_results)
    """
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)
    
    # Random Forest with balanced class weights
    base_model = RandomForestClassifier(
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    # Hyperparameter grid (using RandomizedSearchCV for efficiency)
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    scoring = create_scoring_dict()
    
    # Randomized search with cross-validation
    random_search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit='f1',
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    print("Performing randomized search...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV F1-score: {random_search.best_score_:.4f}")
    
    return best_model, best_params, random_search.cv_results_


def train_xgboost(X_train, y_train, X_val=None, y_val=None, cv=5, random_state=42, n_iter=50):
    """
    Train XGBoost model with hyperparameter tuning and early stopping.
    
    **Improvements for XGBoost Performance:**
    - Increased n_iter from 20 to 50 for better hyperparameter exploration
    - Added early stopping support (if validation set provided)
    - Refined hyperparameter grid (lower learning rates, reduced max_depth)
    - Better regularization parameters
    
    Args:
        X_train: Training features (preprocessed, no scaling needed)
        y_train: Training target
        X_val: Validation features (optional, for early stopping)
        y_val: Validation target (optional)
        cv: Cross-validation folds
        random_state: Random seed
        n_iter: Number of iterations for randomized search (increased to 50)
    
    Returns:
        tuple: (best_model, best_params, cv_results) or None if XGBoost unavailable
    """
    if not XGBOOST_AVAILABLE:
        print("\nSkipping XGBoost (not available)")
        return None, None, None
    
    print("\n" + "="*60)
    print("Training XGBoost")
    print("="*60)
    
    # Convert labels to numeric for XGBoost (requires 0/1)
    y_train_numeric = (y_train == 'Yes').astype(int)
    
    # XGBoost with scale_pos_weight for class imbalance
    # Calculate scale_pos_weight = (negative samples) / (positive samples)
    pos_count = (y_train == 'Yes').sum()
    neg_count = (y_train == 'No').sum()
    scale_pos_weight = neg_count / pos_count
    
    # Prepare validation set if provided
    y_val_numeric = None
    eval_set = None
    early_stopping_rounds = None
    if X_val is not None and y_val is not None:
        y_val_numeric = (y_val == 'Yes').astype(int)
        eval_set = [(X_val, y_val_numeric)]
        early_stopping_rounds = 10
    
    # Base model for RandomizedSearchCV (without early stopping)
    # Early stopping will be applied when retraining the best model
    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
        # Note: early_stopping_rounds not set here - will be added when retraining
    )
    
    # Expanded and refined hyperparameter grid
    # Focus on parameters that improve recall and F1-score
    param_dist = {
        'n_estimators': [200, 300, 400, 500, 600],  # Increased range
        'max_depth': [3, 4, 5, 6, 7],  # Reduced max to prevent overfitting
        'learning_rate': [0.01, 0.05, 0.1, 0.15],  # Lower learning rates for better convergence
        'subsample': [0.8, 0.85, 0.9, 0.95],  # More granular
        'colsample_bytree': [0.8, 0.85, 0.9, 0.95],
        'min_child_weight': [1, 2, 3],  # Reduced range
        'gamma': [0, 0.1, 0.2],  # Reduced range
        'reg_alpha': [0, 0.01, 0.1],  # Lower regularization
        'reg_lambda': [1, 1.5, 2]
    }
    
    scoring = create_scoring_dict()
    
    # Randomized search with increased iterations
    random_search = RandomizedSearchCV(
        base_model,
        param_dist,
        n_iter=n_iter,  # Increased from 20 to 50
        cv=cv,
        scoring=scoring,
        refit='f1',  # Focus on F1-score which balances precision and recall
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    
    print("Performing randomized search with improved hyperparameters...")
    # Note: RandomizedSearchCV doesn't support eval_set directly
    # Early stopping will be applied when refitting the best model
    random_search.fit(X_train, y_train_numeric)
    
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    
    # If validation set provided, retrain best model with early stopping
    if eval_set is not None and early_stopping_rounds is not None:
        print("\nRetraining best model with early stopping on validation set...")
        # Create new model with best parameters and early stopping
        best_model_with_early_stop = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            early_stopping_rounds=early_stopping_rounds
        )
        best_model_with_early_stop.fit(
            X_train, y_train_numeric,
            eval_set=eval_set,
            verbose=False
        )
        best_model = best_model_with_early_stop
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best CV F1-score: {random_search.best_score_:.4f}")
    
    return best_model, best_params, random_search.cv_results_


def train_all_models(X_train, y_train, X_val=None, y_val=None, scale_numerical=True, cv=5, random_state=42):
    """
    Train all models and return results.
    
    Args:
        X_train: Training features (raw dataframe, will be preprocessed)
        y_train: Training target
        X_val: Validation features (optional, for early stopping in XGBoost)
        y_val: Validation target (optional)
        scale_numerical: Whether to scale numerical features
        cv: Cross-validation folds
        random_state: Random seed
    
    Returns:
        dict: Dictionary with model names as keys and (model, params, cv_results) as values
    """
    from src.preprocessing import fit_preprocessor, transform_data
    
    # Preprocess training data
    print("\nPreprocessing training data...")
    preprocessor = fit_preprocessor(X_train, scale_numerical=scale_numerical)
    X_train_processed = transform_data(preprocessor, X_train)
    
    # Preprocess validation data if provided
    X_val_processed = None
    if X_val is not None:
        X_val_processed = transform_data(preprocessor, X_val)
    
    models = {}
    
    # Train Logistic Regression (needs scaling)
    if scale_numerical:
        lr_model, lr_params, lr_cv = train_logistic_regression(
            X_train_processed, y_train, cv=cv, random_state=random_state
        )
        models['Logistic Regression'] = {
            'model': lr_model,
            'params': lr_params,
            'cv_results': lr_cv,
            'preprocessor': preprocessor
        }
    
    # Re-preprocess without scaling for tree models
    preprocessor_unscaled = fit_preprocessor(X_train, scale_numerical=False)
    X_train_unscaled = transform_data(preprocessor_unscaled, X_train)
    
    # Preprocess validation set if provided
    X_val_unscaled = None
    if X_val is not None:
        X_val_unscaled = transform_data(preprocessor_unscaled, X_val)
    
    # Train Random Forest
    rf_model, rf_params, rf_cv = train_random_forest(
        X_train_unscaled, y_train, cv=cv, random_state=random_state
    )
    models['Random Forest'] = {
        'model': rf_model,
        'params': rf_params,
        'cv_results': rf_cv,
        'preprocessor': preprocessor_unscaled
    }
    
    # Train XGBoost (can use validation set for early stopping)
    xgb_model, xgb_params, xgb_cv = train_xgboost(
        X_train_unscaled, y_train,
        X_val=X_val_unscaled, y_val=y_val,
        cv=cv, random_state=random_state, n_iter=50  # Increased iterations
    )
    if xgb_model is not None:
        models['XGBoost'] = {
            'model': xgb_model,
            'params': xgb_params,
            'cv_results': xgb_cv,
            'preprocessor': preprocessor_unscaled
        }
    
    return models


def save_model(model, preprocessor, model_name, output_dir=None):
    """
    Save trained model and preprocessor.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        model_name: Name for the model file
        output_dir: Directory to save model. If None, uses default location.
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "models"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model and preprocessor together
    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump({
        'model': model,
        'preprocessor': preprocessor
    }, model_path)
    
    print(f"Saved model to {model_path}")
    return model_path


if __name__ == "__main__":
    # Test model training
    from data_processing import load_and_process_data
    
    print("Testing model training...")
    X_train, X_test, y_train, y_test = load_and_process_data(save_splits_flag=False)
    
    # Train all models
    models = train_all_models(X_train, y_train, scale_numerical=True)
    
    print(f"\nTrained {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")

