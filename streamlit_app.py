"""
Professional Streamlit Dashboard for Customer Churn Prediction.

A production-ready web application for:
- Individual customer churn prediction
- Batch customer analysis
- Business insights and visualizations
- Model performance metrics
- Feature importance analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.preprocessing import transform_data
from src.feature_engineering import create_all_features


# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    .prediction-high {
        color: #e74c3c;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .prediction-low {
        color: #2ecc71;
        font-weight: bold;
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model():
    """Load the trained model."""
    model_path = project_root / "models" / "best_model.joblib"
    if not model_path.exists():
        return None, None
    
    model_data = joblib.load(model_path)
    return model_data['model'], model_data['preprocessor']


@st.cache_data
def load_sample_data():
    """Load sample data for reference."""
    data_path = project_root / "data" / "raw" / "telco.csv"
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


def predict_churn(customer_data, model, preprocessor):
    """Predict churn probability for a customer."""
    try:
        # Convert to DataFrame if needed
        if isinstance(customer_data, dict):
            customer_df = pd.DataFrame([customer_data])
        else:
            customer_df = customer_data.copy()
        
        # Create engineered features (if model was trained with them)
        # This is safe - if features already exist, they'll be recalculated
        try:
            customer_df = create_all_features(customer_df)
        except Exception as fe_error:
            # If feature engineering fails, continue without it
            # (model might not use engineered features)
            pass
        
        # Transform customer data
        customer_processed = transform_data(preprocessor, customer_df)
        
        # Get prediction
        churn_prob = model.predict_proba(customer_processed)[0, 1]
        churn_pred = model.predict(customer_processed)[0]
        
        return churn_prob, churn_pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")
        return None, None


def create_customer_input_form():
    """Create simplified form with only critical features."""
    st.subheader("📝 Customer Information")
    st.info("💡 **Quick Input:** Only the most important features are shown. Other features use default values.")
    
    # Main features (most important for churn prediction)
    col1, col2 = st.columns(2)
    
    with col1:
        contract = st.selectbox(
            "Contract Type *", 
            ["Month-to-month", "One year", "Two year"],
            help="Most important predictor - month-to-month has highest churn risk"
        )
        tenure = st.slider(
            "Tenure (months) *", 
            0, 72, 12,
            help="New customers (0-12 months) have higher churn risk"
        )
        internet_service = st.selectbox(
            "Internet Service *",
            ["DSL", "Fiber optic", "No"],
            help="Fiber optic customers without support have high churn risk"
        )
        payment_method = st.selectbox(
            "Payment Method *",
            ["Electronic check", "Mailed check", 
             "Bank transfer (automatic)", "Credit card (automatic)"],
            help="Electronic check users have higher churn risk"
        )
    
    with col2:
        tech_support = st.selectbox(
            "Tech Support *",
            ["Yes", "No", "No internet service"],
            help="Customers without tech support have higher churn risk"
        )
        online_security = st.selectbox(
            "Online Security *",
            ["Yes", "No", "No internet service"],
            help="Important for churn prediction"
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($) *", 
            0.0, 200.0, 50.0, 0.1,
            help="Monthly service charges"
        )
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            help="Optional: Senior citizens with high charges have higher risk"
        )
    
    # Calculate TotalCharges
    total_charges = monthly_charges * tenure
    
    # Advanced options (less important features)
    with st.expander("🔧 Advanced Options (Optional - Default values will be used if not set)"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender_adv")
            partner = st.selectbox("Partner", ["Yes", "No"], key="partner_adv")
            dependents = st.selectbox("Dependents", ["Yes", "No"], key="dependents_adv")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="phone_adv")
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], key="multiple_adv")
        
        with col2:
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], key="backup_adv")
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], key="device_adv")
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], key="tv_adv")
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], key="movies_adv")
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="paperless_adv")
    
    # Return all features (use defaults for advanced if not set)
    return {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">📊 Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, preprocessor = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please run `python main.py` first to train the model.")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "🔮 Predict Churn", "📈 Business Insights", "📊 Model Performance", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        st.header("Welcome to Churn Prediction System")
        st.markdown("""
        ### Transform Customer Retention with AI-Powered Predictions
        
        This dashboard provides:
        - **Real-time churn predictions** for individual customers
        - **Business insights** from model analysis
        - **Performance metrics** and model evaluation
        - **Actionable recommendations** for retention strategies
        
        ---
        """)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Load actual metrics
        comparison_path = project_root / "reports" / "model_comparison.csv"
        data_summary_path = project_root / "reports" / "eda_tables" / "data_summary.csv"
        
        accuracy_val = "N/A"
        customer_count = "N/A"
        
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            if len(comparison) > 0 and 'Accuracy' in comparison.columns:
                accuracy_val = f"{comparison.iloc[0]['Accuracy']:.1%}"
        
        if data_summary_path.exists():
            summary = pd.read_csv(data_summary_path)
            total_row = summary[summary['Metric'] == 'Total Rows']
            if len(total_row) > 0:
                customer_count = f"{int(total_row.iloc[0]['Value']):,} Customers"
        
        with col1:
            st.metric("Model Accuracy", accuracy_val, "Actual")
        
        with col2:
            st.metric("Churn Detection", "High Recall", "Critical")
        
        with col3:
            st.metric("Business Impact", "Data-Driven", "Measurable")
        
        with col4:
            st.metric("Data Coverage", customer_count, "Comprehensive")
        
        st.markdown("---")
        
        # Quick start
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Navigate to **"Predict Churn"** to analyze individual customers
        2. View **"Business Insights"** for key churn drivers
        3. Check **"Model Performance"** for detailed metrics
        """)
    
    elif page == "🔮 Predict Churn":
        st.header("🔮 Customer Churn Prediction")
        
        # Customer input form
        customer_data = create_customer_input_form()
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔍 Predict Churn Risk", type="primary"):
            # Convert to DataFrame
            customer_df = pd.DataFrame([customer_data])
            
            # Predict
            with st.spinner("Analyzing customer data..."):
                churn_prob, churn_pred = predict_churn(customer_df, model, preprocessor)
            
            if churn_prob is not None:
                # Display prediction
                st.markdown("### Prediction Result")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Churn Probability",
                        f"{churn_prob:.1%}",
                        delta=f"{'High Risk' if churn_prob > 0.5 else 'Low Risk'}"
                    )
                
                with col2:
                    risk_level = "🔴 HIGH RISK" if churn_prob > 0.5 else "🟢 LOW RISK"
                    st.metric("Risk Level", risk_level)
                
                # Visual indicator
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = churn_prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Risk Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgreen"},
                            {'range': [50, 75], 'color': "yellow"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if churn_prob > 0.5:
                    st.warning("⚠️ **High Churn Risk Detected**")
                    st.markdown("""
                    **Immediate Actions:**
                    - Contact customer for retention offer
                    - Review service quality issues
                    - Offer contract upgrade incentives
                    - Provide additional support services
                    """)
                else:
                    st.success("✅ **Low Churn Risk**")
                    st.markdown("""
                    **Maintenance Actions:**
                    - Continue current service level
                    - Monitor for any changes
                    - Proactive engagement opportunities
                    """)
                
                # Key factors
                st.markdown("### 🔑 Key Risk Factors")
                risk_factors = []
                
                if customer_data['Contract'] == 'Month-to-month':
                    risk_factors.append("⚠️ Month-to-month contract (high churn risk)")
                if customer_data['tenure'] < 12:
                    risk_factors.append("⚠️ Low tenure (new customer risk)")
                if customer_data['TechSupport'] == 'No':
                    risk_factors.append("⚠️ No tech support service")
                if customer_data['PaymentMethod'] == 'Electronic check':
                    risk_factors.append("⚠️ Electronic check payment method")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.info("✅ No major risk factors identified")
    
    elif page == "📈 Business Insights":
        st.header("📈 Business Insights & Analytics")
        
        # Load feature importance
        feature_imp_path = project_root / "reports" / "feature_importance.csv"
        if feature_imp_path.exists():
            feature_imp = pd.read_csv(feature_imp_path)
            
            st.subheader("Top Churn Drivers")
            
            # Top 10 features
            top_features = feature_imp.head(10)
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance (Top 10)",
                labels={'importance': 'Importance Score', 'feature': 'Feature'},
                color='importance',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.subheader("Feature Importance Table")
            st.dataframe(top_features, use_container_width=True)
        
        # Churn rates by feature
        st.subheader("Churn Rates by Key Features")
        
        churn_rates_path = project_root / "reports" / "eda_tables" / "churn_rates_by_feature.csv"
        if churn_rates_path.exists():
            churn_rates = pd.read_csv(churn_rates_path)
            
            # Filter for key features
            key_features = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport']
            filtered_rates = churn_rates[churn_rates['Feature'].isin(key_features)]
            
            for feature in key_features:
                feature_data = filtered_rates[filtered_rates['Feature'] == feature].sort_values('Churn_Rate_Percentage', ascending=False)
                
                if len(feature_data) > 0:
                    fig = px.bar(
                        feature_data,
                        x='Category',
                        y='Churn_Rate_Percentage',
                        title=f"Churn Rate by {feature}",
                        labels={'Churn_Rate_Percentage': 'Churn Rate (%)', 'Category': 'Category'},
                        color='Churn_Rate_Percentage',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Business recommendations
        st.subheader("💼 Actionable Business Recommendations")
        st.markdown("""
        ### 1. Contract Strategy
        - **Focus:** Month-to-month customers have highest churn risk
        - **Action:** Offer incentives for longer contracts (1-year, 2-year)
        - **Expected Impact:** Potential for churn reduction (requires A/B testing to measure)
        
        ### 2. Customer Onboarding
        - **Focus:** New customers (tenure < 12 months) are at risk
        - **Action:** Enhanced onboarding, early engagement programs
        - **Expected Impact:** Potential improvement in retention (requires measurement)
        
        ### 3. Support Services
        - **Focus:** Customers without TechSupport churn more
        - **Action:** Proactively offer support services, free trials
        - **Expected Impact:** Potential for churn reduction (requires A/B testing)
        
        ### 4. Payment Methods
        - **Focus:** Electronic check users have higher churn
        - **Action:** Incentivize automatic payment methods
        - **Expected Impact:** Potential improvement in retention (requires measurement)
        
        **Note:** Actual impact should be measured through A/B testing and real-world deployment.
        """)
    
    elif page == "📊 Model Performance":
        st.header("📊 Model Performance Metrics")
        
        # Load model comparison
        comparison_path = project_root / "reports" / "model_comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            
            st.subheader("Model Comparison")
            
            # Metrics comparison
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
            available_metrics = [m for m in metrics if m in comparison.columns]
            
            for metric in available_metrics:
                fig = px.bar(
                    comparison,
                    x='Model',
                    y=metric,
                    title=f"{metric} Comparison",
                    labels={metric: metric, 'Model': 'Model'},
                    color=metric,
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.subheader("Detailed Metrics Table")
            st.dataframe(comparison, use_container_width=True)
            
            # Best model
            if 'F1_Score' in comparison.columns:
                best_model = comparison.loc[comparison['F1_Score'].idxmax()]
                st.success(f"🏆 **Best Model:** {best_model['Model']} (F1-Score: {best_model['F1_Score']:.3f})")
        
        # Metric explanations
        st.subheader("📚 Metric Explanations")
        st.markdown("""
        - **Accuracy:** Overall correctness of predictions
        - **Precision:** Of predicted churners, how many actually churn
        - **Recall:** Of actual churners, how many we catch (critical for business)
        - **F1-Score:** Balance between precision and recall
        - **ROC-AUC:** Overall model performance across all thresholds
        """)
    
    elif page == "ℹ️ About":
        st.header("ℹ️ About This System")
        
        st.markdown("""
        ### Customer Churn Prediction System
        
        **Version:** 1.0  
        **Last Updated:** 2024
        
        ---
        
        ### System Overview
        
        This machine learning system predicts customer churn with high accuracy, enabling
        proactive retention strategies and significant business value.
        
        ### Key Features
        
        - ✅ Real-time churn predictions
        - ✅ High recall (catches most churners)
        - ✅ Interpretable predictions (SHAP values)
        - ✅ Business insights and recommendations
        - ✅ Production-ready deployment
        
        ### Technology Stack
        
        - **Models:** XGBoost, Random Forest, Logistic Regression
        - **Framework:** scikit-learn, XGBoost
        - **Dashboard:** Streamlit
        - **Visualization:** Plotly
        
        ### Business Value
        
        - **Predictive Power:** Model identifies at-risk customers with high recall
        - **Actionable Insights:** Feature importance guides retention strategies
        - **Data-Driven:** All recommendations based on actual dataset analysis
        
        **Note:** Actual ROI and business impact require external data (intervention costs,
        retention success rates) and should be measured through A/B testing.
        
        ### Contact & Support
        
        For questions or support, please contact the data science team.
        """)


if __name__ == "__main__":
    main()

