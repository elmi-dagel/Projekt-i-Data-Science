# Customer Churn Prediction using Machine Learning

A production-style machine learning project to predict customer churn for telecom companies using the Telco Customer Churn dataset from Kaggle.

## Project Overview

This project builds a complete end-to-end ML pipeline to:
- Predict which customers are at risk of churning
- Identify the most important drivers of churn
- Provide business insights and recommendations

**Target Variable:** Churn (Yes/No)  
**Problem Type:** Binary Classification  
**Focus Metrics:** Recall and F1-score for churn class

## Project Structure

```
churn_model/
├── README.md
├── requirements.txt
├── main.py                        # Main pipeline orchestrator
├── streamlit_app.py              # Interactive Streamlit dashboard
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── download_data.py          # Kaggle API data download
│   ├── data_processing.py        # Load, clean, split data
│   ├── feature_engineering.py    # Create engineered features
│   ├── preprocessing.py          # Feature preprocessing pipelines
│   ├── eda.py                    # Comprehensive EDA analysis
│   ├── modeling.py               # Model training functions
│   ├── evaluation.py             # Metrics & evaluation utilities
│   ├── explainability.py         # SHAP values & feature importance
│   └── report_generator.py       # Generate comprehensive report
├── data/
│   ├── raw/                      # Raw downloaded data
│   └── processed/                # Processed train/test splits
├── models/                       # Saved trained models
└── reports/                      # All outputs (plots, tables, reports)
    ├── eda_plots/                # EDA visualizations
    ├── eda_tables/               # EDA tables (CSV & HTML)
    ├── shap_plots/               # SHAP visualizations
    ├── full_project_report.md    # Comprehensive project report
    └── *.png, *.csv, *.txt       # Evaluation metrics & plots
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Kaggle API Setup (Optional but Recommended)

To download the dataset programmatically, you can set up Kaggle authentication:

1. **Create Kaggle API Token:**
   - Go to https://www.kaggle.com/account
   - Scroll to "API" section
   - Click "Create New API Token"
   - This downloads `kaggle.json`

2. **Place kaggle.json in the correct location:**
   - **Mac/Linux:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<your-username>\.kaggle\kaggle.json`

3. **Set correct permissions (Mac/Linux only):**
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

**Note:** The project uses `kagglehub` which may work without authentication for public datasets, but authentication is recommended for reliability.

## Running the Pipeline

### Step 1: Download Data

```bash
python -m src.download_data
```

This will:
- Download the Telco Customer Churn dataset from Kaggle
- Extract and save it as `data/raw/telco.csv`
- Skip download if file already exists

### Step 2: Run Complete Pipeline

```bash
python main.py
```

This will execute the complete pipeline:
1. **Download data** (if not already present)
2. **Run comprehensive EDA** - Generate all exploratory analysis tables and plots
3. **Load and process data** - Clean data, create engineered features, split into train/validation/test
4. **Train models** - Logistic Regression, Random Forest, XGBoost (with hyperparameter tuning)
5. **Evaluate models** - Calculate metrics, generate confusion matrices, ROC curves
6. **Advanced evaluation** - Threshold optimization, calibration curves, statistical tests
7. **Generate explainability** - SHAP values and feature importance plots
8. **Generate comprehensive report** - Full project report with all findings
9. **Save best model** - Save to `models/best_model.joblib` for deployment

### Step 3: Launch Streamlit Dashboard (Optional)

```bash
streamlit run streamlit_app.py
```

This launches an interactive web dashboard where you can:
- Predict churn for individual customers
- View business insights and churn drivers
- Explore model performance metrics
- Analyze feature importance

## Output Files

After running the pipeline, you'll find:

- **Models:**
  - `models/best_model.joblib` - Best trained model (with preprocessor) for deployment

- **Data:**
  - `data/raw/telco.csv` - Raw dataset from Kaggle
  - `data/processed/train.csv` - Training set
  - `data/processed/test.csv` - Test set

- **EDA Outputs:**
  - `reports/eda_tables/` - All EDA tables (CSV & color-coded HTML)
    - Data summary, churn rates, correlations, feature importance, etc.
  - `reports/eda_plots/` - All EDA visualizations (PNG)
    - Distributions, heatmaps, bivariate analysis, riskiest profiles, etc.

- **Model Evaluation:**
  - `reports/metrics.txt` - Performance metrics for all models
  - `reports/model_comparison.csv` - Detailed model comparison table
  - `reports/roc_curve.png` - ROC curve comparison
  - `reports/confusion_matrix_*.png` - Confusion matrices for each model
  - `reports/calibration_*.png` - Calibration curves for each model
  - `reports/statistical_comparison.csv` - Statistical significance tests

- **Explainability:**
  - `reports/shap_plots/` - SHAP visualizations for each model
  - `reports/feature_importance.csv` - Feature importance rankings

- **Documentation:**
  - `reports/full_project_report.md` - Comprehensive academic-style report
    - Includes methodology, results, business insights, limitations, future work

## Model Evaluation

The project evaluates models using:
- **Recall** (for churn = Yes) - Primary focus (catches churners)
- **Precision** - Of predicted churners, how many actually churn
- **F1-score** - Primary focus (balances precision and recall)
- **ROC-AUC** - Overall model discrimination ability
- **Accuracy** - Overall correctness
- **Brier Score** - Calibration quality
- **Statistical Tests** - McNemar's test for model comparison

Models are optimized for **recall and F1-score** to identify as many churn customers as possible. The project accepts precision ~0.5 as a trade-off for higher recall, which is justified by the business context where false positives (low-cost interventions) are preferable to false negatives (lost customers).

## Key Hypotheses Tested

- **H1:** Month-to-month contracts → Higher churn ✓ (Confirmed)
- **H2:** High MonthlyCharges → Higher churn ✓ (Confirmed)
- **H3:** Low tenure (new customers) → Higher churn ✓ (Confirmed)
- **H4:** Missing support services → Higher churn ✓ (Confirmed)

All hypotheses were tested and confirmed through comprehensive EDA and statistical analysis. Results are documented in `reports/full_project_report.md`.

## Technologies Used

- **Python 3.8+**
- **scikit-learn** - Machine learning models, preprocessing, evaluation
- **XGBoost** - Gradient boosting model with early stopping
- **SHAP** - Model explainability and feature importance
- **pandas/numpy** - Data manipulation and analysis
- **matplotlib/seaborn** - Static visualizations
- **plotly** - Interactive visualizations
- **streamlit** - Interactive web dashboard
- **scipy** - Statistical tests and analysis
- **kagglehub** - Kaggle dataset download
- **joblib** - Model persistence

## Features

### Comprehensive EDA
- Data quality assessment and cleaning decisions
- Statistical hypothesis testing
- Feature interaction analysis
- Outlier detection
- Distribution comparisons
- Bivariate and multivariate analysis
- Non-linear relationship detection
- Professional color-coded tables and visualizations

### Feature Engineering
- Contract risk scores
- Tenure groups
- Payment method risk scores
- Support services count
- Charge per month (reduces multicollinearity)
- Interaction features (senior + high charge, fiber optic + no support)

### Model Training
- **Logistic Regression** - Interpretable baseline with scaling
- **Random Forest** - Balanced performance and interpretability
- **XGBoost** - Maximum recall with early stopping and hyperparameter tuning
- All models use class balancing for imbalanced data
- Hyperparameter tuning with cross-validation

### Advanced Evaluation
- Threshold optimization for business metrics
- Calibration curves and Brier scores
- Statistical significance testing (McNemar's test)
- Comprehensive metric comparison

### Explainability
- SHAP values for model interpretation
- Feature importance rankings
- Top churn drivers identification

### Documentation
- Comprehensive academic-style report
- All decisions and rationales documented
- Business insights and recommendations
- Limitations and future work

## License

This is a course project for educational purposes.

