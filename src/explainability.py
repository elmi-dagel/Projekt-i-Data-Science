"""
Model explainability using SHAP values and feature importance.

This module:
- Generates SHAP values for model interpretation
- Creates SHAP summary plots (bar plot, beeswarm)
- Extracts feature importance for tree-based models
- Identifies top drivers of churn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None


def get_feature_importance_tree(model, feature_names):
    """
    Extract feature importance from tree-based models.
    
    Args:
        model: Trained tree-based model (Random Forest, XGBoost)
        feature_names: List of feature names after preprocessing
    
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return None
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df


def plot_feature_importance(importance_df, model_name, top_n=20, output_path=None):
    """
    Plot feature importance for tree-based models.
    
    Args:
        importance_df: DataFrame with feature and importance columns
        model_name: Name of the model
        top_n: Number of top features to display
        output_path: Path to save plot. If None, uses default location.
    """
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "reports" / "shap_plots" / f"feature_importance_{model_name.replace(' ', '_')}.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved feature importance plot to {output_path}")


def save_feature_importance_csv(importance_df, model_name, output_path=None):
    """
    Save feature importance to CSV for report.
    
    Args:
        importance_df: DataFrame with feature and importance columns
        model_name: Name of the model
        output_path: Path to save CSV. If None, uses default location.
    """
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "reports" / f"feature_importance_{model_name.replace(' ', '_')}.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add rank column
    importance_df = importance_df.copy()
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df = importance_df[['rank', 'feature', 'importance']]
    
    importance_df.to_csv(output_path, index=False)
    print(f"Saved feature importance table to {output_path}")


def calculate_shap_values(model, preprocessor, X_train, X_test, model_name, max_samples=100, random_state=42):
    """
    Calculate SHAP values for model interpretation.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        X_train: Training features (for background data)
        X_test: Test features (for explanation)
        model_name: Name of the model
        max_samples: Maximum number of samples for SHAP (for efficiency)
    
    Returns:
        tuple: (shap_values, shap_explainer, X_test_processed)
    """
    if not SHAP_AVAILABLE:
        print(f"Skipping SHAP for {model_name} (SHAP not available)")
        return None, None, None
    
    from src.preprocessing import transform_data
    
    print(f"\nCalculating SHAP values for {model_name}...")
    
    # Preprocess data
    X_train_processed = transform_data(preprocessor, X_train)
    X_test_processed = transform_data(preprocessor, X_test)
    
    # Limit samples for efficiency
    if len(X_test_processed) > max_samples:
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(X_test_processed), max_samples, replace=False)
        X_test_sample = X_test_processed[sample_indices]
    else:
        X_test_sample = X_test_processed
    
    # Limit background data for tree explainer (increase to 200-300 for better accuracy)
    background_size = min(300, len(X_train_processed))
    if len(X_train_processed) > background_size:
        np.random.seed(random_state)
        background_indices = np.random.choice(len(X_train_processed), background_size, replace=False)
        X_train_background = X_train_processed[background_indices]
    else:
        X_train_background = X_train_processed
    
    try:
        # Choose explainer based on model type
        if hasattr(model, 'predict_proba'):
            # Tree-based models (Random Forest, XGBoost)
            if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sample)
                # For binary classification, get values for positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class (Churn = 'Yes')
            else:
                # Linear models (Logistic Regression)
                explainer = shap.LinearExplainer(model, X_train_background)
                shap_values = explainer.shap_values(X_test_sample)
                # For binary classification
                if len(shap_values.shape) > 1:
                    shap_values = shap_values[:, 1]  # Positive class
        else:
            print(f"Model {model_name} does not support SHAP explainability")
            return None, None, None
        
        print(f"SHAP values calculated for {len(X_test_sample)} samples")
        return shap_values, explainer, X_test_sample
        
    except Exception as e:
        print(f"Error calculating SHAP values for {model_name}: {e}")
        return None, None, None


def create_shap_summary_plot(shap_values, X_test_processed, feature_names, model_name, output_dir=None):
    """
    Create SHAP summary plots (bar and beeswarm).
    
    Args:
        shap_values: SHAP values array
        X_test_processed: Processed test features
        feature_names: List of feature names
        model_name: Name of the model
        output_dir: Directory to save plots. If None, uses default location.
    """
    if shap_values is None:
        return
    
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "shap_plots"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create SHAP summary object
    shap_summary = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(shap_values)),
        data=X_test_processed,
        feature_names=feature_names
    )
    
    # Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_summary, show=False)
    plt.title(f'SHAP Feature Importance (Mean |SHAP|) - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    bar_path = output_dir / f"shap_bar_{model_name.replace(' ', '_')}.png"
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP bar plot to {bar_path}")
    
    # Beeswarm plot (detailed SHAP values)
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_summary, show=False)
    plt.title(f'SHAP Summary Plot - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    beeswarm_path = output_dir / f"shap_beeswarm_{model_name.replace(' ', '_')}.png"
    plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP beeswarm plot to {beeswarm_path}")


def get_top_churn_drivers(shap_values, feature_names, top_n=10):
    """
    Identify top drivers of churn based on SHAP values.
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
        top_n: Number of top drivers to return
    
    Returns:
        pd.DataFrame: Top drivers with mean absolute SHAP values
    """
    if shap_values is None:
        return None
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create dataframe
    drivers_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    return drivers_df.head(top_n)


def explain_model(model, preprocessor, X_train, X_test, model_name, feature_names=None):
    """
    Complete explainability pipeline for a model.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        X_train: Training features
        X_test: Test features
        model_name: Name of the model
        feature_names: List of feature names (optional, will be inferred if None)
    
    Returns:
        dict: Dictionary with explainability results
    """
    from src.preprocessing import get_feature_names as get_feat_names
    
    # Get feature names if not provided
    if feature_names is None:
        feature_names = get_feat_names(preprocessor, X_train)
    
    results = {
        'model_name': model_name,
        'feature_names': feature_names
    }
    
    # Feature importance for tree models
    if hasattr(model, 'feature_importances_'):
        importance_df = get_feature_importance_tree(model, feature_names)
        if importance_df is not None:
            results['feature_importance'] = importance_df
            plot_feature_importance(importance_df, model_name)
            save_feature_importance_csv(importance_df, model_name)
            print(f"\nTop 10 Features for {model_name}:")
            print(importance_df.head(10).to_string(index=False))
    
    # SHAP values
    shap_values, explainer, X_test_sample = calculate_shap_values(
        model, preprocessor, X_train, X_test, model_name
    )
    
    if shap_values is not None:
        results['shap_values'] = shap_values
        results['shap_explainer'] = explainer
        
        # Create SHAP plots
        create_shap_summary_plot(shap_values, X_test_sample, feature_names, model_name)
        
        # Get top drivers
        top_drivers = get_top_churn_drivers(shap_values, feature_names)
        if top_drivers is not None:
            results['top_drivers'] = top_drivers
            print(f"\nTop 10 Churn Drivers for {model_name} (by SHAP):")
            print(top_drivers.to_string(index=False))
    
    return results


def explain_all_models(models_dict, X_train, X_test):
    """
    Generate explainability for all models.
    
    **Business Insights from Feature Importance:**
    - Identifies which customer characteristics drive churn predictions
    - Enables targeted intervention strategies
    - Helps prioritize retention efforts
    - Provides actionable business recommendations
    
    Args:
        models_dict: Dictionary with trained models
        X_train: Training features
        X_test: Test features
    
    Returns:
        dict: Dictionary with explainability results for each model
    """
    print("\n" + "="*60)
    print("Generating Model Explanations")
    print("="*60)
    
    explanations = {}
    
    for model_name, model_info in models_dict.items():
        print(f"\nExplaining {model_name}...")
        explanation = explain_model(
            model_info['model'],
            model_info['preprocessor'],
            X_train,
            X_test,
            model_name
        )
        explanations[model_name] = explanation
    
    # Save combined feature importance for best model
    best_model_name = max(explanations.keys(), 
                         key=lambda x: explanations[x].get('feature_importance', 
                         pd.DataFrame({'importance': [0]})).iloc[0]['importance'] if 'feature_importance' in explanations[x] and len(explanations[x]['feature_importance']) > 0 else 0)
    
    if 'feature_importance' in explanations[best_model_name]:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "reports" / "feature_importance.csv"
        explanations[best_model_name]['feature_importance'].to_csv(output_path, index=False)
        print(f"\nSaved combined feature importance to {output_path}")
    
    return explanations


if __name__ == "__main__":
    # Test explainability
    print("Explainability module test - requires trained models")
    print("Run main.py to test the full pipeline")

