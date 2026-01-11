"""
Main pipeline for Customer Churn Prediction ML Project.

This script orchestrates the entire workflow:
1. Download data (if not exists)
2. Run comprehensive EDA (generate all tables and plots)
3. Load and process data (with documented cleaning decisions)
4. Train models (Logistic Regression, Random Forest, XGBoost)
5. Evaluate models
6. Generate explainability plots
7. Generate comprehensive project report
8. Save best model
"""

import sys
import logging
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.download_data import download_dataset
from src.eda import run_comprehensive_eda
from src.data_processing import load_and_process_data
from src.modeling import train_all_models, save_model
from src.evaluation import generate_all_evaluations, select_best_model
from src.explainability import explain_all_models
from src.report_generator import generate_report


def setup_logging():
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main pipeline execution."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("Customer Churn Prediction ML Pipeline")
    logger.info("="*60)
    
    # Step 1: Download data
    logger.info("\n[Step 1/9] Downloading data...")
    try:
        download_dataset()
        logger.info("Data download completed")
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        sys.exit(1)
    
    # Step 2: Run comprehensive EDA
    logger.info("\n[Step 2/9] Running comprehensive EDA...")
    try:
        eda_results = run_comprehensive_eda()
        logger.info("EDA completed - all tables and plots saved to reports/")
    except Exception as e:
        logger.error(f"EDA failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Load and process data
    logger.info("\n[Step 3/9] Loading and processing data...")
    try:
        # Use validation set for better XGBoost training with early stopping
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_process_data(
            save_splits_flag=True,
            use_feature_engineering=True,  # Use engineered features
            use_validation=True  # Create validation set for XGBoost early stopping
        )
        logger.info(f"Data loaded: {len(X_train)} train, {len(X_val)} validation, {len(X_test)} test samples")
        logger.info(f"Features after engineering: {len(X_train.columns)} columns")
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        sys.exit(1)
    
    # Step 4: Train models
    logger.info("\n[Step 4/9] Training models...")
    try:
        models_dict = train_all_models(
            X_train,
            y_train,
            X_val=X_val,  # Pass validation set for XGBoost early stopping
            y_val=y_val,
            scale_numerical=True,  # For Logistic Regression
            cv=5,
            random_state=42
        )
        logger.info(f"Trained {len(models_dict)} models")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Evaluate models
    logger.info("\n[Step 5/9] Evaluating models...")
    try:
        results, best_model_name, best_result = generate_all_evaluations(
            models_dict,
            X_test,
            y_test
        )
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best F1-score: {best_result['metrics']['f1_score']:.4f}")
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5.5: Advanced evaluation (threshold optimization, calibration, statistical tests)
    logger.info("\n[Step 5.5/9] Advanced model evaluation...")
    try:
        from src.evaluation import optimize_threshold, check_calibration, compare_models_statistically
        
        # Threshold optimization for best model
        if best_result['y_pred_proba'] is not None:
            optimal_threshold = optimize_threshold(
                y_test, 
                best_result['y_pred_proba'], 
                metric='f1', 
                min_recall=0.75
            )
            logger.info(f"Optimal threshold for {best_model_name}: {optimal_threshold:.3f}")
        
        # Calibration check for all models
        logger.info("Checking model calibration...")
        for model_name, result in results.items():
            if result['y_pred_proba'] is not None:
                calibration_metrics = check_calibration(
                    y_test, 
                    result['y_pred_proba'], 
                    model_name
                )
                logger.info(f"  {model_name} - Brier Score: {calibration_metrics['brier_score']:.4f}")
        
        # Statistical comparison between models
        logger.info("Performing statistical comparison between models...")
        stat_comparison = compare_models_statistically(results, y_test, test_type='mcnemar')
        logger.info("Statistical comparison completed")
    except Exception as e:
        logger.warning(f"Advanced evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        logger.warning("Continuing without advanced evaluation...")
    
    # Step 6: Generate explainability
    logger.info("\n[Step 6/9] Generating model explanations...")
    try:
        explanations = explain_all_models(models_dict, X_train, X_test)
        logger.info("Explainability analysis completed")
    except Exception as e:
        logger.warning(f"Explainability generation failed: {e}")
        logger.warning("Continuing without explainability plots...")
        explanations = {}
    
    # Step 7: Generate comprehensive report
    logger.info("\n[Step 7/9] Generating comprehensive project report...")
    try:
        report_path = generate_report()
        logger.info(f"Report generated: {report_path}")
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
        logger.warning("Continuing without report...")
        import traceback
        traceback.print_exc()
    
    # Step 8: Save best model
    logger.info("\n[Step 8/9] Saving best model...")
    try:
        best_model_info = models_dict[best_model_name]
        model_path = save_model(
            best_model_info['model'],
            best_model_info['preprocessor'],
            'best_model',
            project_root / "models"
        )
        logger.info(f"Best model saved to {model_path}")
    except Exception as e:
        logger.error(f"Model saving failed: {e}")
        sys.exit(1)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Pipeline Completed Successfully!")
    logger.info("="*60)
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"Performance Metrics:")
    metrics = best_result['metrics']
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1-score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    logger.info(f"\nOutputs saved to:")
    logger.info(f"  - Model: {project_root / 'models' / 'best_model.joblib'}")
    logger.info(f"  - EDA tables: {project_root / 'reports' / 'eda_tables'}")
    logger.info(f"  - EDA plots: {project_root / 'reports' / 'eda_plots'}")
    logger.info(f"  - Metrics: {project_root / 'reports' / 'metrics.txt'}")
    logger.info(f"  - Model comparison: {project_root / 'reports' / 'model_comparison.csv'}")
    logger.info(f"  - Feature importance: {project_root / 'reports' / 'feature_importance.csv'}")
    logger.info(f"  - SHAP plots: {project_root / 'reports' / 'shap_plots'}")
    logger.info(f"  - Comprehensive report: {project_root / 'reports' / 'full_project_report.md'}")
    logger.info(f"\n🚀 To launch Streamlit dashboard: streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()

