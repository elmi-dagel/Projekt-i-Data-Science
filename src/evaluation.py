"""
Model evaluation and metrics calculation.

This module:
- Calculates classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Generates confusion matrices
- Creates ROC curves
- Saves metrics and plots to reports/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy import stats
from pathlib import Path


def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class (for ROC-AUC)
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label='Yes'),
        'recall': recall_score(y_true, y_pred, pos_label='Yes'),
        'f1_score': f1_score(y_true, y_pred, pos_label='Yes')
    }
    
    # ROC-AUC requires probabilities
    if y_pred_proba is not None:
        # Convert 'Yes'/'No' to binary for ROC-AUC
        y_true_binary = (y_true == 'Yes').astype(int)
        metrics['roc_auc'] = roc_auc_score(y_true_binary, y_pred_proba)
    else:
        metrics['roc_auc'] = None
    
    return metrics


def evaluate_model(model, preprocessor, X_test, y_test, model_name):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        X_test: Test features (raw dataframe)
        y_test: Test target
        model_name: Name of the model
    
    Returns:
        dict: Dictionary with predictions, probabilities, and metrics
    """
    from src.preprocessing import transform_data
    
    # Preprocess test data
    X_test_processed = transform_data(preprocessor, X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_processed)
    
    # Convert numeric predictions (0/1) back to 'No'/'Yes' if needed
    # This handles XGBoost which returns 0/1 instead of 'No'/'Yes'
    if isinstance(y_pred[0], (int, np.integer)) or (isinstance(y_pred[0], np.ndarray) and y_pred[0].dtype in [np.int32, np.int64]):
        y_pred = np.where(y_pred == 1, 'Yes', 'No')
    
    # Get probabilities for positive class (Churn = 'Yes')
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    else:
        y_pred_proba = None
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    return {
        'model_name': model_name,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'metrics': metrics
    }


def evaluate_all_models(models_dict, X_test, y_test):
    """
    Evaluate all trained models.
    
    Args:
        models_dict: Dictionary with model names as keys and dicts with 'model' and 'preprocessor' as values
        X_test: Test features
        y_test: Test target
    
    Returns:
        dict: Dictionary with evaluation results for each model
    """
    results = {}
    
    print("\n" + "="*60)
    print("Evaluating Models on Test Set")
    print("="*60)
    
    for model_name, model_info in models_dict.items():
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(
            model_info['model'],
            model_info['preprocessor'],
            X_test,
            y_test,
            model_name
        )
        results[model_name] = result
        
        # Print metrics
        metrics = result['metrics']
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc'] is not None:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return results


def create_confusion_matrix_plot(y_true, y_pred, model_name, output_path=None):
    """
    Create and save confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        output_path: Path to save plot. If None, uses default location.
    """
    cm = confusion_matrix(y_true, y_pred, labels=['No', 'Yes'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn'],
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "reports" / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved confusion matrix to {output_path}")


def create_roc_curve_plot(results_dict, y_test, output_path=None):
    """
    Create ROC curve plot comparing all models.
    
    Args:
        results_dict: Dictionary with evaluation results
        y_test: True labels
        output_path: Path to save plot. If None, uses default location.
    """
    plt.figure(figsize=(10, 8))
    
    y_true_binary = (y_test == 'Yes').astype(int)
    
    for model_name, result in results_dict.items():
        if result['y_pred_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_true_binary, result['y_pred_proba'])
            roc_auc = result['metrics']['roc_auc']
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "reports" / "roc_curve.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ROC curve to {output_path}")


def save_model_comparison_csv(results_dict, output_path=None):
    """
    Save model comparison as CSV table for report.
    
    Args:
        results_dict: Dictionary with evaluation results
        output_path: Path to save CSV. If None, uses default location.
    """
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "reports" / "model_comparison.csv"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comparison_data = []
    for model_name, result in results_dict.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1_Score': metrics['f1_score'],
            'ROC_AUC': metrics['roc_auc'] if metrics['roc_auc'] is not None else np.nan
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
    comparison_df.to_csv(output_path, index=False)
    print(f"Saved model comparison table to {output_path}")


def save_metrics(results_dict, output_path=None):
    """
    Save evaluation metrics to a text file.
    
    Args:
        results_dict: Dictionary with evaluation results
        output_path: Path to save metrics. If None, uses default location.
    """
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "reports" / "metrics.txt"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Model Evaluation Metrics\n")
        f.write("="*60 + "\n\n")
        
        for model_name, result in results_dict.items():
            metrics = result['metrics']
            f.write(f"{model_name}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-score:  {metrics['f1_score']:.4f}\n")
            if metrics['roc_auc'] is not None:
                f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
            f.write("\n")
        
        # Find best model based on F1-score
        best_model = max(
            results_dict.items(),
            key=lambda x: x[1]['metrics']['f1_score']
        )
        f.write("="*60 + "\n")
        f.write(f"Best Model (by F1-score): {best_model[0]}\n")
        f.write(f"F1-score: {best_model[1]['metrics']['f1_score']:.4f}\n")
        f.write(f"Recall: {best_model[1]['metrics']['recall']:.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"Saved metrics to {output_path}")
    
    # Also save CSV version
    save_model_comparison_csv(results_dict)


def select_best_model(results_dict, metric='f1_score'):
    """
    Select the best model based on a specified metric.
    
    Args:
        results_dict: Dictionary with evaluation results
        metric: Metric to use for selection (default: 'f1_score')
    
    Returns:
        tuple: (best_model_name, best_result)
    """
    best_model = max(
        results_dict.items(),
        key=lambda x: x[1]['metrics'][metric]
    )
    
    return best_model[0], best_model[1]


def generate_all_evaluations(models_dict, X_test, y_test, output_dir=None):
    """
    Generate all evaluation outputs (metrics, plots).
    
    **Metric Selection Rationale:**
    - **Recall:** Critical for churn prediction - we want to catch as many churners as possible
      (minimize False Negatives). Missing a churner is costly.
    - **F1-Score:** Balances precision and recall. Primary metric for model selection.
    - **Precision:** Important to avoid false alarms (False Positives) that waste resources.
    - **ROC-AUC:** Overall model performance across all thresholds.
    - **Accuracy:** Less important due to class imbalance, but included for completeness.
    
    Args:
        models_dict: Dictionary with trained models
        X_test: Test features
        y_test: Test target
        output_dir: Directory for outputs. If None, uses default location.
    
    Returns:
        tuple: (results_dict, best_model_name, best_result)
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate all models
    results = evaluate_all_models(models_dict, X_test, y_test)
    
    # Save metrics (this also saves CSV)
    save_metrics(results, output_dir / "metrics.txt")
    
    # Create ROC curve
    create_roc_curve_plot(results, y_test, output_dir / "roc_curve.png")
    
    # Create confusion matrices for all models
    for model_name, result in results.items():
        create_confusion_matrix_plot(
            y_test,
            result['y_pred'],
            model_name,
            output_dir / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        )
    
    # Select best model
    best_model_name, best_result = select_best_model(results)
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"{'='*60}")
    print(f"F1-score: {best_result['metrics']['f1_score']:.4f}")
    print(f"Recall:   {best_result['metrics']['recall']:.4f}")
    print(f"Precision: {best_result['metrics']['precision']:.4f}")
    if best_result['metrics']['roc_auc'] is not None:
        print(f"ROC-AUC:  {best_result['metrics']['roc_auc']:.4f}")
    
    return results, best_model_name, best_result


def optimize_threshold(y_true, y_pred_proba, metric='f1', min_recall=0.75):
    """
    Find optimal threshold for classification.
    
    Args:
        y_true: True labels ('Yes'/'No')
        y_pred_proba: Predicted probabilities for positive class
        metric: Metric to optimize ('f1', 'recall', 'precision')
        min_recall: Minimum recall requirement (only used if metric='f1')
    
    Returns:
        float: Optimal threshold value
    """
    # Convert to binary
    y_true_binary = (y_true == 'Yes').astype(int)
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_proba)
    
    if metric == 'f1':
        # Find threshold that meets min_recall and maximizes F1
        valid_indices = np.where(recall >= min_recall)[0]
        if len(valid_indices) > 0:
            f1_scores = 2 * (precision[valid_indices] * recall[valid_indices]) / \
                       (precision[valid_indices] + recall[valid_indices] + 1e-10)
            optimal_idx = valid_indices[np.argmax(f1_scores)]
            return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    elif metric == 'recall':
        # Maximize recall while maintaining reasonable precision (>0.5)
        valid_indices = np.where(precision >= 0.5)[0]
        if len(valid_indices) > 0:
            optimal_idx = valid_indices[np.argmax(recall[valid_indices])]
            return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    elif metric == 'precision':
        # Maximize precision while maintaining reasonable recall (>0.5)
        valid_indices = np.where(recall >= 0.5)[0]
        if len(valid_indices) > 0:
            optimal_idx = valid_indices[np.argmax(precision[valid_indices])]
            return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return 0.5  # Default threshold


def check_calibration(y_true, y_pred_proba, model_name, output_path=None):
    """
    Check model calibration and calculate Brier score.
    
    Args:
        y_true: True labels ('Yes'/'No')
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        output_path: Path to save calibration plot
    
    Returns:
        dict: Calibration metrics including Brier score
    """
    y_true_binary = (y_true == 'Yes').astype(int)
    
    # Calculate Brier score
    brier = brier_score_loss(y_true_binary, y_pred_proba)
    
    # Calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_binary, y_pred_proba, n_bins=10
    )
    
    # Create plot
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "reports" / f"calibration_{model_name.replace(' ', '_')}.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label=model_name, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=1)
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Curve - {model_name}\nBrier Score: {brier:.4f}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved calibration curve to {output_path}")
    print(f"Brier Score: {brier:.4f} (lower is better, 0 = perfect calibration)")
    
    return {
        'brier_score': brier,
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }


def compare_models_statistically(results_dict, y_test, test_type='mcnemar'):
    """
    Perform statistical significance testing between models.
    
    Args:
        results_dict: Dictionary with evaluation results
        y_test: True labels
        test_type: 'mcnemar' or 'paired_t'
    
    Returns:
        dict: Statistical test results
    """
    y_true_binary = (y_test == 'Yes').astype(int)
    
    # Get model predictions
    model_predictions = {}
    for model_name, result in results_dict.items():
        # Convert predictions to binary
        pred_binary = (result['y_pred'] == 'Yes').astype(int)
        model_predictions[model_name] = pred_binary
    
    # Perform pairwise comparisons
    model_names = list(model_predictions.keys())
    comparison_results = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1_name = model_names[i]
            model2_name = model_names[j]
            
            pred1 = model_predictions[model1_name]
            pred2 = model_predictions[model2_name]
            
            if test_type == 'mcnemar':
                # McNemar's test (for paired binary predictions)
                # Create contingency table
                both_correct = ((pred1 == y_true_binary) & (pred2 == y_true_binary)).sum()
                both_wrong = ((pred1 != y_true_binary) & (pred2 != y_true_binary)).sum()
                model1_correct = ((pred1 == y_true_binary) & (pred2 != y_true_binary)).sum()
                model2_correct = ((pred1 != y_true_binary) & (pred2 == y_true_binary)).sum()
                
                # McNemar's test statistic
                if model1_correct + model2_correct > 0:
                    chi2_stat = ((abs(model1_correct - model2_correct) - 1) ** 2) / (model1_correct + model2_correct)
                    # p-value from chi-square distribution with 1 degree of freedom
                    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
                else:
                    chi2_stat = 0
                    p_value = 1.0
                
                comparison_results.append({
                    'Model_1': model1_name,
                    'Model_2': model2_name,
                    'Test_Type': 'McNemar',
                    'Statistic': chi2_stat,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Model1_Correct_Only': model1_correct,
                    'Model2_Correct_Only': model2_correct
                })
            
            elif test_type == 'paired_t':
                # Paired t-test on prediction accuracies (if we had multiple runs)
                # For single run, we can't do paired t-test
                # This would require cross-validation scores
                comparison_results.append({
                    'Model_1': model1_name,
                    'Model_2': model2_name,
                    'Test_Type': 'Paired_t',
                    'Note': 'Requires cross-validation scores (not available for single run)',
                    'P_Value': np.nan,
                    'Significant': 'N/A'
                })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Save results
    project_root = Path(__file__).parent.parent
    output_path = project_root / "reports" / "statistical_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    print(f"\nSaved statistical comparison to {output_path}")
    
    return comparison_df


if __name__ == "__main__":
    # Test evaluation
    print("Evaluation module test - requires trained models")
    print("Run main.py to test the full pipeline")

