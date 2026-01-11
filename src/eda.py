"""
Comprehensive Exploratory Data Analysis (EDA) for Customer Churn Prediction.

This module performs detailed EDA including:
- Data overview and quality assessment
- Target variable analysis
- Numerical and categorical feature analysis
- Outlier detection
- Hypothesis testing
- Feature relationship analysis

All results are saved as tables (CSV) and visualizations (PNG) for report inclusion.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def format_dataframe_for_csv(df):
    """
    Formatera DataFrame för bättre läsbarhet i CSV-filer.
    Avrundar procent, formaterar numeriska värden, etc.
    
    Args:
        df: DataFrame att formatera
    
    Returns:
        DataFrame: Formaterad DataFrame
    """
    df_formatted = df.copy()
    
    for col in df_formatted.columns:
        # Formatera procent-kolumner (avrunda till 2 decimaler)
        if 'percentage' in col.lower() or 'rate' in col.lower() or 'pct' in col.lower():
            if df_formatted[col].dtype in ['float64', 'float32']:
                df_formatted[col] = df_formatted[col].round(2)
        
        # Formatera P-values (avrunda till 4 decimaler eller scientific notation)
        elif 'p_value' in col.lower() or 'pvalue' in col.lower():
            if df_formatted[col].dtype in ['float64', 'float32']:
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:.4e}" if pd.notna(x) and x < 0.0001 else round(x, 4) if pd.notna(x) else x
                )
        
        # Formatera statistiska värden (Chi2, F-statistic, etc.) - avrunda till 2 decimaler
        elif any(stat in col.lower() for stat in ['chi2', 'f_statistic', 'cramers', 'mutual', 'importance_score']):
            if df_formatted[col].dtype in ['float64', 'float32']:
                df_formatted[col] = df_formatted[col].round(2)
        
        # Formatera andra numeriska kolumner
        elif df_formatted[col].dtype in ['float64', 'float32']:
            # Avrunda till 2 decimaler för små tal, 1 decimal för större
            df_formatted[col] = df_formatted[col].apply(
                lambda x: round(x, 2) if pd.notna(x) and abs(x) < 1000 else round(x, 1) if pd.notna(x) else x
            )
    
    return df_formatted


def load_data_for_eda(data_path=None):
    """Load data for EDA analysis."""
    if data_path is None:
        project_root = Path(__file__).parent.parent
        data_path = project_root / "data" / "raw" / "telco.csv"
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def data_overview(df, output_dir=None):
    """
    Generate data overview and quality assessment.
    
    Returns:
        dict: Summary statistics and quality metrics
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("DATA OVERVIEW & QUALITY ASSESSMENT")
    print("="*60)
    
    overview = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Create summary DataFrame
    summary_data = {
        'Metric': [
            'Total Rows',
            'Total Columns',
            'Memory Usage (MB)',
            'Duplicate Rows',
            'Missing Values (Total)'
        ],
        'Value': [
            df.shape[0],
            df.shape[1],
            f"{overview['memory_usage_mb']:.2f}",
            overview['duplicate_rows'],
            df.isnull().sum().sum()
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "data_summary.csv", index=False)
    print(f"\nSaved data summary to {output_dir / 'data_summary.csv'}")
    
    # Missing values report
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values,
        'Data_Type': df.dtypes.values
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
    
    if len(missing_df) > 0:
        missing_df.to_csv(output_dir / "missing_values_report.csv", index=False)
        print(f"Saved missing values report to {output_dir / 'missing_values_report.csv'}")
        print("\nMissing Values Found:")
        print(missing_df)
    else:
        # Check for empty strings in TotalCharges (common issue)
        if 'TotalCharges' in df.columns:
            empty_strings = (df['TotalCharges'] == ' ').sum() if df['TotalCharges'].dtype == 'object' else 0
            if empty_strings > 0:
                print(f"\n⚠️  Found {empty_strings} empty strings in TotalCharges column")
                print("   This will be handled during data cleaning.")
    
    # Data types report
    dtype_df = pd.DataFrame({
        'Column': df.columns,
        'Data_Type': df.dtypes.values,
        'Unique_Values': [df[col].nunique() for col in df.columns],
        'Sample_Values': [str(list(df[col].dropna().head(3).values)) for col in df.columns]
    })
    dtype_df.to_csv(output_dir / "data_types_report.csv", index=False)
    
    print(f"\nData Overview:")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Memory: {overview['memory_usage_mb']:.2f} MB")
    print(f"  Duplicates: {overview['duplicate_rows']}")
    
    return overview


def target_variable_analysis(df, output_dir=None):
    """Analyze target variable (Churn) distribution."""
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("TARGET VARIABLE ANALYSIS (CHURN)")
    print("="*60)
    
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    
    churn_df = pd.DataFrame({
        'Churn_Status': churn_counts.index,
        'Count': churn_counts.values,
        'Percentage': churn_pct.values
    })
    churn_df.to_csv(output_dir / "churn_distribution.csv", index=False)
    print(f"\nSaved churn distribution to {output_dir / 'churn_distribution.csv'}")
    
    print(f"\nChurn Distribution:")
    print(churn_df.to_string(index=False))
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    axes[0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                startangle=90, colors=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Churn Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    # Bar chart
    bars = axes[1].bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[1].set_title('Churn Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Churn Status', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/len(df)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "churn_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved churn distribution plot to {plots_dir / 'churn_distribution.png'}")
    
    # Class imbalance assessment
    imbalance_ratio = churn_counts.min() / churn_counts.max()
    print(f"\n⚠️  Class Imbalance Ratio: {imbalance_ratio:.3f}")
    print("   (Ratio < 0.5 indicates significant imbalance)")
    print("   Recommendation: Use class_weight='balanced' or SMOTE")
    
    return churn_df


def numerical_features_analysis(df, output_dir=None):
    """Analyze numerical features."""
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    
    print("\n" + "="*60)
    print("NUMERICAL FEATURES ANALYSIS")
    print("="*60)
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Convert TotalCharges to numeric if needed
    df_analysis = df.copy()
    if df_analysis['TotalCharges'].dtype == 'object':
        df_analysis['TotalCharges'] = df_analysis['TotalCharges'].replace(' ', np.nan)
        df_analysis['TotalCharges'] = pd.to_numeric(df_analysis['TotalCharges'], errors='coerce')
    
    # Descriptive statistics
    stats_df = df_analysis[numerical_cols].describe()
    stats_df = format_dataframe_for_csv(stats_df)
    stats_df.to_csv(output_dir / "numerical_stats.csv")
    print(f"\nSaved numerical statistics to {output_dir / 'numerical_stats.csv'}")
    print("\nDescriptive Statistics:")
    print(stats_df)
    
    # Outlier detection
    outliers_data = []
    for col in numerical_cols:
        Q1 = df_analysis[col].quantile(0.25)
        Q3 = df_analysis[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_analysis[(df_analysis[col] < lower_bound) | (df_analysis[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df_analysis)) * 100
        
        outliers_data.append({
            'Feature': col,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': outlier_pct,
            'Min_Value': df_analysis[col].min(),
            'Max_Value': df_analysis[col].max()
        })
    
    outliers_df = pd.DataFrame(outliers_data)
    outliers_df = format_dataframe_for_csv(outliers_df)
    outliers_df.to_csv(output_dir / "outliers_detected.csv", index=False)
    print(f"\nSaved outlier detection results to {output_dir / 'outliers_detected.csv'}")
    print("\nOutlier Detection (IQR Method):")
    print(outliers_df.to_string(index=False))
    
    # Correlation analysis
    corr_matrix = df_analysis[numerical_cols + ['Churn']].copy()
    corr_matrix['Churn'] = (corr_matrix['Churn'] == 'Yes').astype(int)
    correlation = corr_matrix.corr()
    correlation = format_dataframe_for_csv(correlation)
    correlation.to_csv(output_dir / "correlations.csv")
    print(f"\nSaved correlation matrix to {output_dir / 'correlations.csv'}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Distribution plots
    for idx, col in enumerate(numerical_cols):
        axes[0, idx].hist(df_analysis[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[0, idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
        axes[0, idx].set_xlabel(col)
        axes[0, idx].set_ylabel('Frequency')
    
    # Box plots for outliers
    for idx, col in enumerate(numerical_cols):
        axes[1, idx].boxplot(df_analysis[col].dropna(), vert=True)
        axes[1, idx].set_title(f'{col} Box Plot (Outlier Detection)', fontsize=12, fontweight='bold')
        axes[1, idx].set_ylabel(col)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "numerical_features_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved numerical features plots to {plots_dir / 'numerical_features_analysis.png'}")
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to {plots_dir / 'correlation_heatmap.png'}")
    
    return stats_df, outliers_df, correlation


def categorical_features_analysis(df, output_dir=None):
    """Analyze categorical features and their relationship with churn."""
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    
    print("\n" + "="*60)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*60)
    
    # Exclude customerID and target variable
    categorical_cols = [col for col in df.columns 
                        if col not in ['customerID', 'Churn', 'tenure', 'MonthlyCharges', 'TotalCharges']]
    
    # Distribution of categorical features
    cat_distributions = []
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        for val, count in value_counts.items():
            cat_distributions.append({
                'Feature': col,
                'Category': val,
                'Count': count,
                'Percentage': (count / len(df)) * 100
            })
    
    cat_dist_df = pd.DataFrame(cat_distributions)
    # Formatera procent
    cat_dist_df['Percentage'] = cat_dist_df['Percentage'].round(2)
    cat_dist_df = format_dataframe_for_csv(cat_dist_df)
    cat_dist_df.to_csv(output_dir / "categorical_distributions.csv", index=False)
    print(f"\nSaved categorical distributions to {output_dir / 'categorical_distributions.csv'}")
    
    # Churn rates by feature
    churn_rates_data = []
    for col in categorical_cols:
        churn_by_cat = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
        total_by_cat = df[col].value_counts()
        
        for cat in churn_by_cat.index:
            churn_rates_data.append({
                'Feature': col,
                'Category': cat,
                'Total_Count': total_by_cat[cat],
                'Churn_Count': (df[(df[col] == cat) & (df['Churn'] == 'Yes')]).shape[0],
                'Churn_Rate_Percentage': churn_by_cat[cat]
            })
    
    churn_rates_df = pd.DataFrame(churn_rates_data)
    churn_rates_df = churn_rates_df.sort_values(['Feature', 'Churn_Rate_Percentage'], ascending=[True, False])
    # Formatera churn rates
    churn_rates_df['Churn_Rate_Percentage'] = churn_rates_df['Churn_Rate_Percentage'].round(2)
    churn_rates_df = format_dataframe_for_csv(churn_rates_df)
    churn_rates_df.to_csv(output_dir / "churn_rates_by_feature.csv", index=False)
    print(f"\nSaved churn rates by feature to {output_dir / 'churn_rates_by_feature.csv'}")
    
    # Create pivot tables for better readability - one table per feature
    pivot_tables_dir = output_dir / "churn_rates_pivot_tables"
    pivot_tables_dir.mkdir(exist_ok=True)
    
    for feature in churn_rates_df['Feature'].unique():
        feature_data = churn_rates_df[churn_rates_df['Feature'] == feature].copy()
        feature_data = feature_data.sort_values('Churn_Rate_Percentage', ascending=False)
        
        # Create pivot-style table
        pivot_table = pd.DataFrame({
            'Category': feature_data['Category'].values,
            'Churn Rate (%)': feature_data['Churn_Rate_Percentage'].values,
            'Total Count': feature_data['Total_Count'].values,
            'Churn Count': feature_data['Churn_Count'].values
        })
        
        # Save as CSV
        safe_filename = feature.replace(' ', '_').replace('/', '_')
        pivot_table.to_csv(pivot_tables_dir / f"{safe_filename}_churn_rates.csv", index=False)
    
    print(f"Saved pivot tables to {pivot_tables_dir}")
    
    # Visualizations - Top features by churn rate difference
    top_features = ['Contract', 'PaymentMethod', 'InternetService', 'OnlineSecurity', 'TechSupport']
    
    fig, axes = plt.subplots(len(top_features), 1, figsize=(14, 4*len(top_features)))
    if len(top_features) == 1:
        axes = [axes]
    
    for idx, feature in enumerate(top_features):
        if feature in categorical_cols:
            churn_by_cat = df.groupby(feature)['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
            churn_by_cat = churn_by_cat.sort_values(ascending=False)
            
            bars = axes[idx].bar(range(len(churn_by_cat)), churn_by_cat.values, 
                                color=plt.cm.RdYlGn_r(churn_by_cat.values / 100))
            axes[idx].set_xticks(range(len(churn_by_cat)))
            axes[idx].set_xticklabels(churn_by_cat.index, rotation=45, ha='right')
            axes[idx].set_title(f'Churn Rate by {feature}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Churn Rate (%)')
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, churn_by_cat.values)):
                axes[idx].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                              f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "churn_rates_by_category.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved churn rates visualization to {plots_dir / 'churn_rates_by_category.png'}")
    
    return cat_dist_df, churn_rates_df


def hypothesis_testing(df, output_dir=None):
    """Test the key churn hypotheses."""
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
    
    output_dir = Path(output_dir)
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    # Convert TotalCharges if needed
    df_test = df.copy()
    if df_test['TotalCharges'].dtype == 'object':
        df_test['TotalCharges'] = df_test['TotalCharges'].replace(' ', np.nan)
        df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'], errors='coerce')
    
    df_test['Churn_Binary'] = (df_test['Churn'] == 'Yes').astype(int)
    
    hypotheses_results = []
    
    # H1: Month-to-month contracts → Higher churn
    month_to_month_churn = df_test[df_test['Contract'] == 'Month-to-month']['Churn_Binary'].mean()
    longer_contract_churn = df_test[df_test['Contract'] != 'Month-to-month']['Churn_Binary'].mean()
    
    # Statistical test (chi-square)
    contingency_table = pd.crosstab(df_test['Contract'] == 'Month-to-month', df_test['Churn_Binary'])
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    hypotheses_results.append({
        'Hypothesis': 'H1: Month-to-month contracts → Higher churn',
        'Month_to_Month_Churn_Rate': month_to_month_churn * 100,
        'Longer_Contract_Churn_Rate': longer_contract_churn * 100,
        'Difference': (month_to_month_churn - longer_contract_churn) * 100,
        'Chi2_Statistic': chi2,
        'P_Value': p_value,
        'Result': 'Supported' if p_value < 0.05 and month_to_month_churn > longer_contract_churn else 'Not Supported'
    })
    
    # H2: High MonthlyCharges → Higher churn
    median_charges = df_test['MonthlyCharges'].median()
    high_charges_churn = df_test[df_test['MonthlyCharges'] > median_charges]['Churn_Binary'].mean()
    low_charges_churn = df_test[df_test['MonthlyCharges'] <= median_charges]['Churn_Binary'].mean()
    
    # T-test
    high_charges_group = df_test[df_test['MonthlyCharges'] > median_charges]['Churn_Binary']
    low_charges_group = df_test[df_test['MonthlyCharges'] <= median_charges]['Churn_Binary']
    t_stat, p_value = stats.ttest_ind(high_charges_group, low_charges_group)
    
    hypotheses_results.append({
        'Hypothesis': 'H2: High MonthlyCharges → Higher churn',
        'High_Charges_Churn_Rate': high_charges_churn * 100,
        'Low_Charges_Churn_Rate': low_charges_churn * 100,
        'Difference': (high_charges_churn - low_charges_churn) * 100,
        'T_Statistic': t_stat,
        'P_Value': p_value,
        'Result': 'Supported' if p_value < 0.05 and high_charges_churn > low_charges_churn else 'Not Supported'
    })
    
    # H3: Low tenure → Higher churn
    median_tenure = df_test['tenure'].median()
    low_tenure_churn = df_test[df_test['tenure'] <= median_tenure]['Churn_Binary'].mean()
    high_tenure_churn = df_test[df_test['tenure'] > median_tenure]['Churn_Binary'].mean()
    
    low_tenure_group = df_test[df_test['tenure'] <= median_tenure]['Churn_Binary']
    high_tenure_group = df_test[df_test['tenure'] > median_tenure]['Churn_Binary']
    t_stat, p_value = stats.ttest_ind(low_tenure_group, high_tenure_group)
    
    hypotheses_results.append({
        'Hypothesis': 'H3: Low tenure → Higher churn',
        'Low_Tenure_Churn_Rate': low_tenure_churn * 100,
        'High_Tenure_Churn_Rate': high_tenure_churn * 100,
        'Difference': (low_tenure_churn - high_tenure_churn) * 100,
        'T_Statistic': t_stat,
        'P_Value': p_value,
        'Result': 'Supported' if p_value < 0.05 and low_tenure_churn > high_tenure_churn else 'Not Supported'
    })
    
    # H4: Missing support services → Higher churn
    has_support = df_test[df_test['TechSupport'].isin(['Yes'])]['Churn_Binary'].mean()
    no_support = df_test[df_test['TechSupport'].isin(['No', 'No internet service'])]['Churn_Binary'].mean()
    
    contingency_table = pd.crosstab(df_test['TechSupport'].isin(['Yes']), df_test['Churn_Binary'])
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    hypotheses_results.append({
        'Hypothesis': 'H4: Missing support services → Higher churn',
        'With_Support_Churn_Rate': has_support * 100,
        'Without_Support_Churn_Rate': no_support * 100,
        'Difference': (no_support - has_support) * 100,
        'Chi2_Statistic': chi2,
        'P_Value': p_value,
        'Result': 'Supported' if p_value < 0.05 and no_support > has_support else 'Not Supported'
    })
    
    hypotheses_df = pd.DataFrame(hypotheses_results)
    hypotheses_df = format_dataframe_for_csv(hypotheses_df)
    hypotheses_df.to_csv(output_dir / "hypothesis_test_results.csv", index=False)
    print(f"\nSaved hypothesis test results to {output_dir / 'hypothesis_test_results.csv'}")
    print("\nHypothesis Testing Results:")
    print(hypotheses_df.to_string(index=False))
    
    return hypotheses_df


def create_color_coded_table(df, column_to_color, output_path, color_scheme='RdYlGn_r', title='Table'):
    """
    Create color-coded HTML table for reports.
    
    Args:
        df: DataFrame to style
        column_to_color: Column name to apply color gradient
        output_path: Path to save HTML file
        color_scheme: Color scheme ('RdYlGn_r', 'Blues', 'Reds', etc.)
        title: Table title
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create styled DataFrame
    styled_df = df.style.background_gradient(
        subset=[column_to_color],
        cmap=color_scheme,
        axis=0
    ).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#4472C4'), 
                                     ('color', 'white'),
                                     ('font-weight', 'bold'),
                                     ('text-align', 'center'),
                                     ('padding', '8px')]},
        {'selector': 'td', 'props': [('text-align', 'center'),
                                     ('padding', '6px')]},
        {'selector': 'table', 'props': [('border-collapse', 'collapse'),
                                        ('width', '100%'),
                                        ('border', '1px solid #ddd')]}
    ]).set_caption(title)
    
    # Save as HTML
    html = styled_df.to_html()
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"<html><head><title>{title}</title></head><body>{html}</body></html>")
    
    # Also save CSV
    csv_path = output_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)


def create_professional_heatmap(data, title, output_path, cmap='coolwarm', annot=True, fmt='.2f'):
    """
    Create professional heatmap with proper color scheme.
    
    Args:
        data: 2D array or DataFrame for heatmap
        title: Plot title
        output_path: Path to save plot
        cmap: Color map ('coolwarm', 'RdYlGn_r', etc.)
        annot: Whether to annotate cells
        fmt: Format string for annotations
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0 if cmap == 'coolwarm' else None,
        square=False,
        linewidths=1,
        cbar_kws={"shrink": 0.8, "label": "Value"},
        vmin=data.min().min() if hasattr(data, 'min') else None,
        vmax=data.max().max() if hasattr(data, 'max') else None
    )
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def distribution_comparison_by_churn(df, output_dir=None):
    """
    Compare distributions of numerical features by churn status.
    
    Creates overlay histograms, KDE plots, and statistical tests.
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("DISTRIBUTION COMPARISON BY CHURN")
    print("="*60)
    
    # Prepare data
    df_analysis = df.copy()
    if df_analysis['TotalCharges'].dtype == 'object':
        df_analysis['TotalCharges'] = df_analysis['TotalCharges'].replace(' ', np.nan)
        df_analysis['TotalCharges'] = pd.to_numeric(df_analysis['TotalCharges'], errors='coerce')
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    comparison_results = []
    
    # Create comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    colors = {'Yes': '#e74c3c', 'No': '#2ecc71'}  # Red for churn, Green for no churn
    
    for idx, col in enumerate(numerical_cols):
        churn_yes = df_analysis[df_analysis['Churn'] == 'Yes'][col].dropna()
        churn_no = df_analysis[df_analysis['Churn'] == 'No'][col].dropna()
        
        # Histogram overlay
        axes[idx, 0].hist(churn_no, bins=30, alpha=0.6, label='No Churn', 
                         color=colors['No'], edgecolor='black')
        axes[idx, 0].hist(churn_yes, bins=30, alpha=0.6, label='Churn', 
                          color=colors['Yes'], edgecolor='black')
        axes[idx, 0].set_title(f'{col} Distribution by Churn', fontsize=12, fontweight='bold')
        axes[idx, 0].set_xlabel(col)
        axes[idx, 0].set_ylabel('Frequency')
        axes[idx, 0].legend()
        axes[idx, 0].grid(alpha=0.3)
        
        # KDE plot
        sns.kdeplot(data=df_analysis, x=col, hue='Churn', ax=axes[idx, 1], 
                   palette=colors, linewidth=2)
        axes[idx, 1].set_title(f'{col} KDE by Churn', fontsize=12, fontweight='bold')
        axes[idx, 1].set_xlabel(col)
        axes[idx, 1].set_ylabel('Density')
        axes[idx, 1].grid(alpha=0.3)
        
        # Statistical tests
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(churn_yes, churn_no)
        
        # Mann-Whitney U test
        u_stat, u_p = stats.mannwhitneyu(churn_yes, churn_no, alternative='two-sided')
        
        # Quantile comparison
        q1_yes, q2_yes, q3_yes = churn_yes.quantile([0.25, 0.5, 0.75])
        q1_no, q2_no, q3_no = churn_no.quantile([0.25, 0.5, 0.75])
        
        comparison_results.append({
            'Feature': col,
            'Churn_Mean': churn_yes.mean(),
            'No_Churn_Mean': churn_no.mean(),
            'Churn_Median': q2_yes,
            'No_Churn_Median': q2_no,
            'Churn_Q1': q1_yes,
            'Churn_Q3': q3_yes,
            'No_Churn_Q1': q1_no,
            'No_Churn_Q3': q3_no,
            'KS_Statistic': ks_stat,
            'KS_PValue': ks_p,
            'MannWhitney_U': u_stat,
            'MannWhitney_PValue': u_p,
            'Significant_Difference': 'Yes' if ks_p < 0.05 else 'No'
        })
    
    plt.tight_layout()
    plt.savefig(plots_dir / "distributions_by_churn.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution comparison plot to {plots_dir / 'distributions_by_churn.png'}")
    
    # Save results table
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_dir / "distribution_comparison_by_churn.csv", index=False)
    create_color_coded_table(
        comparison_df,
        'KS_PValue',
        output_dir / "distribution_comparison_by_churn.html",
        color_scheme='RdYlGn_r',
        title='Distribution Comparison by Churn Status'
    )
    print(f"Saved distribution comparison table to {output_dir / 'distribution_comparison_by_churn.csv'}")
    
    return comparison_df


def feature_interactions_analysis(df, output_dir=None):
    """
    Analyze feature interactions and their impact on churn.
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    
    print("\n" + "="*60)
    print("FEATURE INTERACTIONS ANALYSIS")
    print("="*60)
    
    # Prepare data
    df_analysis = df.copy()
    if df_analysis['TotalCharges'].dtype == 'object':
        df_analysis['TotalCharges'] = df_analysis['TotalCharges'].replace(' ', np.nan)
        df_analysis['TotalCharges'] = pd.to_numeric(df_analysis['TotalCharges'], errors='coerce')
    
    # Create tenure groups
    df_analysis['tenure_group'] = pd.cut(
        df_analysis['tenure'],
        bins=[-1, 12, 24, 48, 100],
        labels=['New (0-12)', 'Short (13-24)', 'Medium (25-48)', 'Long (49+)']
    )
    
    # Create MonthlyCharges groups
    df_analysis['MonthlyCharges_group'] = pd.qcut(
        df_analysis['MonthlyCharges'],
        q=3,
        labels=['Low', 'Medium', 'High'],
        duplicates='drop'
    )
    
    # Count support services
    support_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df_analysis['support_count'] = df_analysis[support_cols].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    
    interactions = []
    
    # Contract × Tenure groups
    for contract in df_analysis['Contract'].unique():
        for tenure_group in df_analysis['tenure_group'].unique():
            subset = df_analysis[(df_analysis['Contract'] == contract) & 
                                (df_analysis['tenure_group'] == tenure_group)]
            if len(subset) > 0:
                churn_rate = (subset['Churn'] == 'Yes').mean() * 100
                interactions.append({
                    'Interaction': f'Contract × Tenure',
                    'Category_1': contract,
                    'Category_2': str(tenure_group),
                    'Sample_Size': len(subset),
                    'Churn_Rate': churn_rate,
                    'Churn_Count': (subset['Churn'] == 'Yes').sum()
                })
    
    # PaymentMethod × Contract
    for payment in df_analysis['PaymentMethod'].unique():
        for contract in df_analysis['Contract'].unique():
            subset = df_analysis[(df_analysis['PaymentMethod'] == payment) & 
                                (df_analysis['Contract'] == contract)]
            if len(subset) > 0:
                churn_rate = (subset['Churn'] == 'Yes').mean() * 100
                interactions.append({
                    'Interaction': f'PaymentMethod × Contract',
                    'Category_1': payment,
                    'Category_2': contract,
                    'Sample_Size': len(subset),
                    'Churn_Rate': churn_rate,
                    'Churn_Count': (subset['Churn'] == 'Yes').sum()
                })
    
    # SeniorCitizen × MonthlyCharges groups
    for senior in [0, 1]:
        for charge_group in df_analysis['MonthlyCharges_group'].unique():
            subset = df_analysis[(df_analysis['SeniorCitizen'] == senior) & 
                                (df_analysis['MonthlyCharges_group'] == charge_group)]
            if len(subset) > 0:
                churn_rate = (subset['Churn'] == 'Yes').mean() * 100
                interactions.append({
                    'Interaction': f'SeniorCitizen × MonthlyCharges',
                    'Category_1': 'Senior' if senior == 1 else 'Non-Senior',
                    'Category_2': str(charge_group),
                    'Sample_Size': len(subset),
                    'Churn_Rate': churn_rate,
                    'Churn_Count': (subset['Churn'] == 'Yes').sum()
                })
    
    # InternetService × Support services count
    for internet in df_analysis['InternetService'].unique():
        for support_count in sorted(df_analysis['support_count'].unique()):
            subset = df_analysis[(df_analysis['InternetService'] == internet) & 
                                (df_analysis['support_count'] == support_count)]
            if len(subset) > 0:
                churn_rate = (subset['Churn'] == 'Yes').mean() * 100
                interactions.append({
                    'Interaction': f'InternetService × Support_Count',
                    'Category_1': internet,
                    'Category_2': f'{int(support_count)} services',
                    'Sample_Size': len(subset),
                    'Churn_Rate': churn_rate,
                    'Churn_Count': (subset['Churn'] == 'Yes').sum()
                })
    
    interactions_df = pd.DataFrame(interactions)
    interactions_df = interactions_df.sort_values('Churn_Rate', ascending=False)
    
    # Formatera
    interactions_df['Churn_Rate'] = interactions_df['Churn_Rate'].round(2)
    interactions_df = format_dataframe_for_csv(interactions_df)
    
    # Save CSV
    interactions_df.to_csv(output_dir / "feature_interactions.csv", index=False)
    
    # Create color-coded HTML table
    create_color_coded_table(
        interactions_df,
        'Churn_Rate',
        output_dir / "feature_interactions.html",
        color_scheme='RdYlGn_r',
        title='Feature Interactions - Churn Rates'
    )
    
    # Create heatmap for Contract × Tenure
    contract_tenure = interactions_df[interactions_df['Interaction'] == 'Contract × Tenure']
    if len(contract_tenure) > 0:
        pivot_data = contract_tenure.pivot_table(
            values='Churn_Rate',
            index='Category_1',
            columns='Category_2',
            aggfunc='mean'
        )
        create_professional_heatmap(
            pivot_data,
            'Churn Rate: Contract × Tenure Groups',
            plots_dir / "feature_interactions_heatmap.png",
            cmap='RdYlGn_r'
        )
        print(f"Saved feature interactions heatmap to {plots_dir / 'feature_interactions_heatmap.png'}")
    
    print(f"Saved feature interactions to {output_dir / 'feature_interactions.csv'}")
    return interactions_df


def identify_riskiest_profiles(df, output_dir=None, top_n=20):
    """
    Identify top riskiest customer profiles based on feature combinations.
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    
    print("\n" + "="*60)
    print("RISKIEST CUSTOMER PROFILES")
    print("="*60)
    
    # Prepare data
    df_analysis = df.copy()
    if df_analysis['TotalCharges'].dtype == 'object':
        df_analysis['TotalCharges'] = df_analysis['TotalCharges'].replace(' ', np.nan)
        df_analysis['TotalCharges'] = pd.to_numeric(df_analysis['TotalCharges'], errors='coerce')
    
    # Create tenure groups
    df_analysis['tenure_group'] = pd.cut(
        df_analysis['tenure'],
        bins=[-1, 12, 24, 48, 100],
        labels=['New', 'Short', 'Medium', 'Long']
    )
    
    # Count support services
    support_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df_analysis['support_count'] = df_analysis[support_cols].apply(
        lambda x: (x == 'Yes').sum(), axis=1
    )
    
    # Key feature combinations
    key_features = ['Contract', 'PaymentMethod', 'tenure_group', 'support_count', 
                   'InternetService', 'SeniorCitizen']
    
    profiles = []
    
    # Group by key combinations
    for name, group in df_analysis.groupby(key_features):
        contract, payment, tenure_grp, support_cnt, internet, senior = name
        
        churn_count = (group['Churn'] == 'Yes').sum()
        total_count = len(group)
        churn_rate = (churn_count / total_count) * 100 if total_count > 0 else 0
        
        # Create profile description
        profile_desc = f"{contract} + {payment} + {tenure_grp} tenure + {int(support_cnt)} support"
        if internet != 'No':
            profile_desc += f" + {internet}"
        if senior == 1:
            profile_desc += " + Senior"
        
        # Risk level categorization
        if churn_rate > 45:
            risk_level = 'Very High Risk'
        elif churn_rate > 35:
            risk_level = 'High Risk'
        elif churn_rate > 25:
            risk_level = 'Medium-High Risk'
        elif churn_rate > 15:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'Low Risk'
        
        profiles.append({
            'Profile': profile_desc,
            'Contract': contract,
            'PaymentMethod': payment,
            'Tenure_Group': str(tenure_grp),
            'Support_Count': int(support_cnt),
            'InternetService': internet,
            'SeniorCitizen': senior,
            'Total_Customers': total_count,
            'Churn_Count': churn_count,
            'Churn_Rate': churn_rate,
            'Risk_Level': risk_level
        })
    
    profiles_df = pd.DataFrame(profiles)
    profiles_df = profiles_df.sort_values('Churn_Rate', ascending=False).head(top_n)
    
    # Add rank
    profiles_df.insert(0, 'Rank', range(1, len(profiles_df) + 1))
    
    # Formatera
    profiles_df['Churn_Rate'] = profiles_df['Churn_Rate'].round(2)
    profiles_df = format_dataframe_for_csv(profiles_df)
    
    # Save CSV
    profiles_df.to_csv(output_dir / "riskiest_customer_profiles.csv", index=False)
    
    # Create color-coded HTML table
    create_color_coded_table(
        profiles_df,
        'Churn_Rate',
        output_dir / "riskiest_customer_profiles.html",
        color_scheme='RdYlGn_r',
        title='Top Riskiest Customer Profiles'
    )
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    colors_map = {
        'Very High Risk': '#c0392b',
        'High Risk': '#e74c3c',
        'Medium-High Risk': '#f39c12',
        'Medium Risk': '#f1c40f',
        'Low Risk': '#2ecc71'
    }
    bar_colors = [colors_map.get(risk, '#95a5a6') for risk in profiles_df['Risk_Level']]
    
    bars = ax.barh(range(len(profiles_df)), profiles_df['Churn_Rate'], color=bar_colors)
    ax.set_yticks(range(len(profiles_df)))
    ax.set_yticklabels([f"Rank {r}" for r in profiles_df['Rank']], fontsize=8)
    ax.set_xlabel('Churn Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Top Riskiest Customer Profiles', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, profiles_df['Churn_Rate'])):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'{rate:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "riskiest_profiles.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved riskiest profiles plot to {plots_dir / 'riskiest_profiles.png'}")
    print(f"Saved riskiest profiles table to {output_dir / 'riskiest_customer_profiles.csv'}")
    
    return profiles_df


def bivariate_analysis(df, output_dir=None):
    """
    Perform bivariate analysis with crosstabs and heatmaps.
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots" / "bivariate_heatmaps"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("BIVARIATE ANALYSIS")
    print("="*60)
    
    # Prepare data
    df_analysis = df.copy()
    if df_analysis['TotalCharges'].dtype == 'object':
        df_analysis['TotalCharges'] = df_analysis['TotalCharges'].replace(' ', np.nan)
        df_analysis['TotalCharges'] = pd.to_numeric(df_analysis['TotalCharges'], errors='coerce')
    
    # Create tenure groups
    df_analysis['tenure_group'] = pd.cut(
        df_analysis['tenure'],
        bins=[-1, 12, 24, 48, 100],
        labels=['New (0-12)', 'Short (13-24)', 'Medium (25-48)', 'Long (49+)']
    )
    
    df_analysis['Churn_Binary'] = (df_analysis['Churn'] == 'Yes').astype(int)
    
    bivariate_results = []
    
    # Feature pairs to analyze
    feature_pairs = [
        ('Contract', 'PaymentMethod'),
        ('Contract', 'tenure_group'),
        ('InternetService', 'TechSupport'),
        ('SeniorCitizen', 'Contract'),
        ('PaymentMethod', 'tenure_group')
    ]
    
    for feat1, feat2 in feature_pairs:
        # Create crosstab
        crosstab = pd.crosstab(
            df_analysis[feat1],
            df_analysis[feat2],
            values=df_analysis['Churn_Binary'],
            aggfunc='mean'
        ) * 100  # Convert to percentage
        
        # Calculate sample sizes
        size_tab = pd.crosstab(df_analysis[feat1], df_analysis[feat2])
        
        # Chi-square test
        contingency = pd.crosstab(df_analysis[feat1], df_analysis[feat2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        # Cramér's V
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        # Save heatmap
        create_professional_heatmap(
            crosstab,
            f'Churn Rate: {feat1} × {feat2}',
            plots_dir / f"{feat1}_{feat2}_heatmap.png",
            cmap='coolwarm',
            fmt='.1f'
        )
        
        # Store results
        for idx in crosstab.index:
            for col in crosstab.columns:
                bivariate_results.append({
                    'Feature_1': feat1,
                    'Feature_2': feat2,
                    'Category_1': idx,
                    'Category_2': col,
                    'Churn_Rate': crosstab.loc[idx, col],
                    'Sample_Size': size_tab.loc[idx, col],
                    'Chi2_Statistic': chi2,
                    'P_Value': p_value,
                    'Cramers_V': cramers_v,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
    
    bivariate_df = pd.DataFrame(bivariate_results)
    
    # Formatera
    bivariate_df['Churn_Rate'] = bivariate_df['Churn_Rate'].round(2)
    bivariate_df = format_dataframe_for_csv(bivariate_df)
    bivariate_df.to_csv(output_dir / "bivariate_analysis.csv", index=False)
    
    # Create pivot tables for better readability - one table per feature pair
    pivot_tables_dir = output_dir / "bivariate_pivot_tables"
    pivot_tables_dir.mkdir(exist_ok=True)
    
    for feat1, feat2 in feature_pairs:
        pair_data = bivariate_df[
            (bivariate_df['Feature_1'] == feat1) & 
            (bivariate_df['Feature_2'] == feat2)
        ].copy()
        
        if len(pair_data) > 0:
            # Create pivot table with Churn_Rate as values
            pivot_table = pair_data.pivot_table(
                values='Churn_Rate',
                index='Category_1',
                columns='Category_2',
                aggfunc='mean',
                fill_value=0
            )
            
            # Also create table with sample sizes
            size_pivot = pair_data.pivot_table(
                values='Sample_Size',
                index='Category_1',
                columns='Category_2',
                aggfunc='mean',
                fill_value=0
            )
            
            # Format pivot table
            pivot_table = format_dataframe_for_csv(pivot_table)
            size_pivot = format_dataframe_for_csv(size_pivot)
            
            # Save churn rate pivot table
            safe_filename = f"{feat1.replace(' ', '_')}_x_{feat2.replace(' ', '_')}"
            pivot_table.to_csv(pivot_tables_dir / f"{safe_filename}_churn_rates.csv")
            
            # Save sample size pivot table
            size_pivot.to_csv(pivot_tables_dir / f"{safe_filename}_sample_sizes.csv")
            
            # Create combined table with both churn rate and sample size
            combined_data = []
            for idx in pivot_table.index:
                for col in pivot_table.columns:
                    combined_data.append({
                        f'{feat1}': idx,
                        f'{feat2}': col,
                        'Churn Rate (%)': pivot_table.loc[idx, col],
                        'Sample Size': int(size_pivot.loc[idx, col]) if idx in size_pivot.index and col in size_pivot.columns else 0
                    })
            
            combined_df = pd.DataFrame(combined_data)
            combined_df = combined_df.sort_values('Churn Rate (%)', ascending=False)
            combined_df.to_csv(pivot_tables_dir / f"{safe_filename}_combined.csv", index=False)
    
    print(f"Saved bivariate pivot tables to {pivot_tables_dir}")
    
    # Create color-coded HTML table
    create_color_coded_table(
        bivariate_df,
        'Churn_Rate',
        output_dir / "bivariate_analysis.html",
        color_scheme='coolwarm',
        title='Bivariate Analysis - Churn Rates'
    )
    
    print(f"Saved bivariate analysis to {output_dir / 'bivariate_analysis.csv'}")
    print(f"Saved heatmaps to {plots_dir}")
    
    return bivariate_df


def detect_nonlinear_relationships(df, output_dir=None):
    """
    Detect non-linear relationships between numerical features and churn.
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    
    print("\n" + "="*60)
    print("NON-LINEAR RELATIONSHIP DETECTION")
    print("="*60)
    
    # Prepare data
    df_analysis = df.copy()
    if df_analysis['TotalCharges'].dtype == 'object':
        df_analysis['TotalCharges'] = df_analysis['TotalCharges'].replace(' ', np.nan)
        df_analysis['TotalCharges'] = pd.to_numeric(df_analysis['TotalCharges'], errors='coerce')
    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    nonlinear_results = []
    
    fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(16, 12))
    colors = {'Yes': '#e74c3c', 'No': '#3498db'}  # Red for churn, Blue for no churn
    
    for idx, col in enumerate(numerical_cols):
        # Scatter plot with churn overlay
        churn_yes = df_analysis[df_analysis['Churn'] == 'Yes']
        churn_no = df_analysis[df_analysis['Churn'] == 'No']
        
        axes[idx, 0].scatter(churn_no[col], np.random.randn(len(churn_no)) * 0.1,
                           alpha=0.3, s=20, c=colors['No'], label='No Churn')
        axes[idx, 0].scatter(churn_yes[col], np.random.randn(len(churn_yes)) * 0.1 + 1,
                           alpha=0.3, s=20, c=colors['Yes'], label='Churn')
        axes[idx, 0].set_xlabel(col, fontsize=11)
        axes[idx, 0].set_ylabel('Churn Status (jittered)', fontsize=11)
        axes[idx, 0].set_title(f'{col} vs Churn (Scatter)', fontsize=12, fontweight='bold')
        axes[idx, 0].legend()
        axes[idx, 0].grid(alpha=0.3)
        
        # Binning analysis
        bins = np.linspace(df_analysis[col].min(), df_analysis[col].max(), 10)
        df_analysis['bin'] = pd.cut(df_analysis[col], bins=bins)
        
        bin_stats = df_analysis.groupby('bin').agg({
            'Churn': lambda x: (x == 'Yes').mean() * 100,
            col: 'mean'
        }).reset_index()
        bin_stats.columns = ['bin', 'churn_rate', 'bin_center']
        
        axes[idx, 1].bar(bin_stats['bin_center'], bin_stats['churn_rate'],
                        width=(bins[1] - bins[0]) * 0.8, alpha=0.7, color='orange', edgecolor='black')
        axes[idx, 1].set_xlabel(col, fontsize=11)
        axes[idx, 1].set_ylabel('Churn Rate (%)', fontsize=11)
        axes[idx, 1].set_title(f'{col} vs Churn (Binned)', fontsize=12, fontweight='bold')
        axes[idx, 1].grid(axis='y', alpha=0.3)
        
        # Store binning results
        for _, row in bin_stats.iterrows():
            nonlinear_results.append({
                'Feature': col,
                'Bin_Center': row['bin_center'],
                'Churn_Rate': row['churn_rate'],
                'Bin_Range': str(row['bin'])
            })
    
    plt.tight_layout()
    plt.savefig(plots_dir / "nonlinear_relationships.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved non-linear relationships plot to {plots_dir / 'nonlinear_relationships.png'}")
    
    # Save results
    nonlinear_df = pd.DataFrame(nonlinear_results)
    nonlinear_df = format_dataframe_for_csv(nonlinear_df)
    nonlinear_df.to_csv(output_dir / "nonlinear_relationships.csv", index=False)
    
    # Create color-coded HTML table
    create_color_coded_table(
        nonlinear_df,
        'Churn_Rate',
        output_dir / "nonlinear_relationships.html",
        color_scheme='RdYlGn_r',
        title='Non-Linear Relationship Detection'
    )
    
    print(f"Saved non-linear relationships table to {output_dir / 'nonlinear_relationships.csv'}")
    
    return nonlinear_df


def calculate_preliminary_importance(df, output_dir=None):
    """
    Calculate preliminary feature importance using univariate methods.
    """
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports" / "eda_tables"
        plots_dir = project_root / "reports" / "eda_plots"
    
    output_dir = Path(output_dir)
    plots_dir = Path(plots_dir)
    
    print("\n" + "="*60)
    print("PRELIMINARY FEATURE IMPORTANCE")
    print("="*60)
    
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data
    df_analysis = df.copy()
    if df_analysis['TotalCharges'].dtype == 'object':
        df_analysis['TotalCharges'] = df_analysis['TotalCharges'].replace(' ', np.nan)
        df_analysis['TotalCharges'] = pd.to_numeric(df_analysis['TotalCharges'], errors='coerce')
        df_analysis['TotalCharges'] = df_analysis['TotalCharges'].fillna(0)
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df_analysis['Churn'])
    
    importance_results = []
    
    # Numerical features - Mutual Information and ANOVA F-test
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        X_num = df_analysis[[col]].fillna(0)
        
        # Mutual Information
        mi_score = mutual_info_classif(X_num, y, random_state=42)[0]
        
        # ANOVA F-test
        from sklearn.feature_selection import f_classif
        f_stat, f_p = f_classif(X_num, y)
        
        importance_results.append({
            'Feature': col,
            'Type': 'Numerical',
            'Mutual_Information': mi_score,
            'F_Statistic': f_stat[0],
            'F_PValue': f_p[0],
            'Importance_Score': mi_score  # Use MI as primary score
        })
    
    # Categorical features - Chi-square test
    categorical_cols = [col for col in df_analysis.columns 
                       if col not in numerical_cols + ['Churn', 'customerID']]
    
    for col in categorical_cols:
        if df_analysis[col].dtype == 'object' or df_analysis[col].dtype.name == 'category':
            # Chi-square test
            contingency = pd.crosstab(df_analysis[col], df_analysis['Churn'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            # Cramér's V as importance score
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            
            importance_results.append({
                'Feature': col,
                'Type': 'Categorical',
                'Chi2_Statistic': chi2,
                'Chi2_PValue': p_value,
                'Cramers_V': cramers_v,
                'Importance_Score': cramers_v  # Use Cramér's V as importance
            })
    
    importance_df = pd.DataFrame(importance_results)
    importance_df = importance_df.sort_values('Importance_Score', ascending=False)
    importance_df.insert(0, 'Rank', range(1, len(importance_df) + 1))
    
    # Formatera
    importance_df = format_dataframe_for_csv(importance_df)
    
    # Save CSV
    importance_df.to_csv(output_dir / "preliminary_feature_importance.csv", index=False)
    
    # Create color-coded HTML table
    create_color_coded_table(
        importance_df,
        'Importance_Score',
        output_dir / "preliminary_feature_importance.html",
        color_scheme='Blues',
        title='Preliminary Feature Importance'
    )
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = importance_df.head(15)
    bars = ax.barh(range(len(top_features)), top_features['Importance_Score'],
                  color=plt.cm.Blues(np.linspace(0.4, 1, len(top_features))))
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Preliminary Feature Importance (Top 15)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_features['Importance_Score'])):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "preliminary_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved preliminary importance plot to {plots_dir / 'preliminary_importance.png'}")
    print(f"Saved preliminary importance table to {output_dir / 'preliminary_feature_importance.csv'}")
    
    return importance_df


def generate_eda_report(df, output_dir=None):
    """Generate text summary of EDA findings."""
    if output_dir is None:
        project_root = Path(__file__).parent.parent
        output_dir = project_root / "reports"
    
    output_dir = Path(output_dir)
    
    report = []
    report.append("="*80)
    report.append("EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    report.append("="*80)
    report.append("")
    
    report.append("1. DATA OVERVIEW")
    report.append("-" * 80)
    report.append(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    report.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    report.append(f"Duplicate Rows: {df.duplicated().sum()}")
    report.append("")
    
    report.append("2. TARGET VARIABLE (CHURN)")
    report.append("-" * 80)
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    report.append(f"Churn Distribution:")
    for status, count in churn_counts.items():
        report.append(f"  {status}: {count} ({churn_pct[status]:.2f}%)")
    report.append(f"\n⚠️  Class Imbalance: Ratio = {churn_counts.min() / churn_counts.max():.3f}")
    report.append("   Recommendation: Address class imbalance in modeling")
    report.append("")
    
    report.append("3. DATA QUALITY ISSUES")
    report.append("-" * 80)
    # Check TotalCharges
    if df['TotalCharges'].dtype == 'object':
        empty_strings = (df['TotalCharges'] == ' ').sum()
        if empty_strings > 0:
            report.append(f"⚠️  TotalCharges: {empty_strings} empty strings found")
            report.append("   Solution: Convert to numeric, set to 0 for tenure=0")
    report.append("")
    
    report.append("4. KEY FINDINGS")
    report.append("-" * 80)
    report.append("- See detailed tables in reports/eda_tables/")
    report.append("- See visualizations in reports/eda_plots/")
    report.append("")
    
    report_text = "\n".join(report)
    
    with open(output_dir / "eda_report.txt", 'w') as f:
        f.write(report_text)
    
    print(f"\nSaved EDA report to {output_dir / 'eda_report.txt'}")
    return report_text


def run_comprehensive_eda(data_path=None):
    """
    Run complete EDA pipeline including enhanced analyses.
    
    Returns:
        dict: All EDA results and dataframes
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*80)
    
    # Load data
    df = load_data_for_eda(data_path)
    
    # Run basic EDA components
    overview = data_overview(df)
    churn_dist = target_variable_analysis(df)
    num_stats, outliers, correlations = numerical_features_analysis(df)
    cat_dist, churn_rates = categorical_features_analysis(df)
    hypotheses = hypothesis_testing(df)
    
    # Run enhanced EDA (Phase 0)
    print("\n" + "="*80)
    print("ENHANCED EDA - FEATURE INTERACTIONS & ADVANCED ANALYSIS")
    print("="*80)
    
    distribution_comp = distribution_comparison_by_churn(df)
    feature_interactions = feature_interactions_analysis(df)
    riskiest_profiles = identify_riskiest_profiles(df)
    bivariate_results = bivariate_analysis(df)
    nonlinear_results = detect_nonlinear_relationships(df)
    preliminary_importance = calculate_preliminary_importance(df)
    
    # Generate report
    eda_report = generate_eda_report(df)
    
    print("\n" + "="*80)
    print("EDA COMPLETE - All results saved to reports/")
    print("="*80)
    
    return {
        'dataframe': df,
        'overview': overview,
        'churn_distribution': churn_dist,
        'numerical_stats': num_stats,
        'outliers': outliers,
        'correlations': correlations,
        'categorical_distributions': cat_dist,
        'churn_rates': churn_rates,
        'hypotheses': hypotheses,
        'distribution_comparison': distribution_comp,
        'feature_interactions': feature_interactions,
        'riskiest_profiles': riskiest_profiles,
        'bivariate_analysis': bivariate_results,
        'nonlinear_relationships': nonlinear_results,
        'preliminary_importance': preliminary_importance
    }


if __name__ == "__main__":
    # Run EDA
    results = run_comprehensive_eda()
    print("\nEDA pipeline completed successfully!")

