"""
CHURN PREDICTION DASHBOARD
Interactive Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

# ============================================
# LOAD MODELS
# ============================================

@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load("churn_model.pkl")
        encoder = joblib.load("encoder.pkl")
        cat_cols = joblib.load("cat_cols.pkl")
        train_features = joblib.load("train_features.pkl")
        return model, encoder, cat_cols, train_features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

model, encoder, cat_cols, TRAIN_FEATURES = load_models()

# Configurations
contract_map = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
payment_risk = {
    'Electronic check': 3,
    'Mailed check': 2,
    'Bank transfer (automatic)': 1,
    'Credit card (automatic)': 1
}

ACTION_COSTS = {
    'Reduce ₹50': 50,
    'Reduce ₹100': 100,
    'Reduce ₹200': 200,
    'Add Online Security': 30,
    'Add Tech Support': 40,
    'Switch to Auto-pay': 10,
    'Upgrade to 1-year Contract': 80
}

# ============================================
# PREDICTION FUNCTIONS
# ============================================

def preprocess_customer(customer_dict):
    """Preprocess customer data"""
    row_df = pd.DataFrame([customer_dict])
    
    # Feature engineering
    row_df['tenure_risk'] = pd.cut(
        row_df['tenure'],
        bins=[-1, 3, 6, 12, 24, 100],
        labels=[5, 4, 3, 2, 1],
        include_lowest=True
    ).astype(int)[0]
    
    row_df['charge_pressure'] = row_df['MonthlyCharges'] / (row_df['tenure'] + 1)
    row_df['relative_price'] = 0
    
    prot_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    row_df['protection_gap'] = (row_df[prot_cols] == 'No').sum(axis=1).values[0]
    
    row_df['contract_risk'] = row_df['Contract'].map(contract_map).values[0]
    row_df['payment_risk'] = row_df['PaymentMethod'].map(payment_risk).values[0]
    
    serv_cols = ['PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies']
    row_df['service_complexity'] = (row_df[serv_cols] != 'No').sum(axis=1).values[0]
    
    row_df['engagement_score'] = row_df['TotalCharges'] / (
        row_df['MonthlyCharges'] * (row_df['tenure'] + 1) + 0.01
    )
    
    return row_df

def predict_churn(customer_dict):
    """Predict churn probability"""
    row_df = preprocess_customer(customer_dict)
    
    num_cols = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'tenure_risk', 'charge_pressure', 'relative_price',
        'protection_gap', 'contract_risk', 'payment_risk',
        'service_complexity', 'engagement_score'
    ]
    
    row_encoded = encoder.transform(row_df[cat_cols])
    encoded_df = pd.DataFrame(
        row_encoded,
        columns=encoder.get_feature_names_out(cat_cols),
        index=row_df.index
    )
    
    final_row = pd.concat([row_df[num_cols], encoded_df], axis=1)
    final_row = final_row.reindex(columns=TRAIN_FEATURES, fill_value=0)
    
    return model.predict_proba(final_row)[0][1]

def generate_recommendations(customer_dict, base_prob):
    """Generate retention recommendations"""
    results = []
    
    # Price reductions
    for reduction in [50, 100, 200]:
        temp = customer_dict.copy()
        temp['MonthlyCharges'] = max(0, temp['MonthlyCharges'] - reduction)
        temp['TotalCharges'] = temp['MonthlyCharges'] * temp['tenure']
        new_prob = predict_churn(temp)
        
        if new_prob < base_prob:
            results.append({
                'Action': f'Reduce ₹{reduction}',
                'New Churn Risk': f"{new_prob:.1%}",
                'Risk Reduction': f"{(base_prob - new_prob):.1%}",
                'Monthly Cost': f"₹{ACTION_COSTS[f'Reduce ₹{reduction}']}",
                'ROI Score': f"{(base_prob - new_prob) / ACTION_COSTS[f'Reduce ₹{reduction}']:.4f}"
            })
    
    # Add services
    if customer_dict.get('OnlineSecurity') == 'No':
        temp = customer_dict.copy()
        temp['OnlineSecurity'] = 'Yes'
        new_prob = predict_churn(temp)
        
        if new_prob < base_prob:
            results.append({
                'Action': 'Add Online Security',
                'New Churn Risk': f"{new_prob:.1%}",
                'Risk Reduction': f"{(base_prob - new_prob):.1%}",
                'Monthly Cost': f"₹{ACTION_COSTS['Add Online Security']}",
                'ROI Score': f"{(base_prob - new_prob) / ACTION_COSTS['Add Online Security']:.4f}"
            })
    
    if customer_dict.get('TechSupport') == 'No':
        temp = customer_dict.copy()
        temp['TechSupport'] = 'Yes'
        new_prob = predict_churn(temp)
        
        if new_prob < base_prob:
            results.append({
                'Action': 'Add Tech Support',
                'New Churn Risk': f"{new_prob:.1%}",
                'Risk Reduction': f"{(base_prob - new_prob):.1%}",
                'Monthly Cost': f"₹{ACTION_COSTS['Add Tech Support']}",
                'ROI Score': f"{(base_prob - new_prob) / ACTION_COSTS['Add Tech Support']:.4f}"
            })
    
    # Payment method
    if customer_dict.get('PaymentMethod') == 'Electronic check':
        temp = customer_dict.copy()
        temp['PaymentMethod'] = 'Credit card (automatic)'
        new_prob = predict_churn(temp)
        
        if new_prob < base_prob:
            results.append({
                'Action': 'Switch to Auto-pay',
                'New Churn Risk': f"{new_prob:.1%}",
                'Risk Reduction': f"{(base_prob - new_prob):.1%}",
                'Monthly Cost': f"₹{ACTION_COSTS['Switch to Auto-pay']}",
                'ROI Score': f"{(base_prob - new_prob) / ACTION_COSTS['Switch to Auto-pay']:.4f}"
            })
    
    # Contract upgrade
    if customer_dict.get('Contract') == 'Month-to-month':
        temp = customer_dict.copy()
        temp['Contract'] = 'One year'
        new_prob = predict_churn(temp)
        
        if new_prob < base_prob:
            results.append({
                'Action': 'Upgrade to 1-year Contract',
                'New Churn Risk': f"{new_prob:.1%}",
                'Risk Reduction': f"{(base_prob - new_prob):.1%}",
                'Monthly Cost': f"₹{ACTION_COSTS['Upgrade to 1-year Contract']}",
                'ROI Score': f"{(base_prob - new_prob) / ACTION_COSTS['Upgrade to 1-year Contract']:.4f}"
            })
    
    return pd.DataFrame(results)

# ============================================
# MAIN APP
# ============================================

st.title("📊 Churn Prediction & Retention System")
st.markdown("### AI-Powered Customer Retention Platform")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["🔮 Single Prediction", "📁 Batch Prediction", "ℹ️ About"]
)

# ============================================
# PAGE 1: SINGLE PREDICTION
# ============================================

if page == "🔮 Single Prediction":
    st.header("Single Customer Churn Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Customer Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    
    with col2:
        st.subheader("📞 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    with col3:
        st.subheader("💳 Billing")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges (₹)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges (₹)", 0.0, 10000.0, float(monthly_charges * tenure))
    
    if st.button("🔮 Predict Churn Risk", type="primary"):
        customer_data = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Predict
        churn_prob = predict_churn(customer_data)
        
        # Display result
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{churn_prob:.1%}")
        
        with col2:
            if churn_prob < 0.3:
                risk_band = "Low Risk 🟢"
                risk_color = "green"
            elif churn_prob < 0.6:
                risk_band = "Medium Risk 🟡"
                risk_color = "orange"
            else:
                risk_band = "High Risk 🔴"
                risk_color = "red"
            
            st.metric("Risk Band", risk_band)
        
        with col3:
            early_warning = "YES ⚠️" if churn_prob >= 0.5 else "NO ✅"
            st.metric("Early Warning", early_warning)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            title={'text': "Churn Risk Score"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        if churn_prob >= 0.5:
            st.subheader("💡 Retention Recommendations")
            recommendations = generate_recommendations(customer_data, churn_prob)
            
            if not recommendations.empty:
                st.dataframe(recommendations, use_container_width=True)
            else:
                st.info("No cost-effective retention strategies available for this customer.")

# ============================================
# PAGE 2: BATCH PREDICTION
# ============================================

elif page == "📁 Batch Prediction":
    st.header("Batch Customer Churn Prediction")
    
    st.markdown("""
    Upload a CSV file with customer data to predict churn for multiple customers at once.
    
    **Required columns:**
    - gender, SeniorCitizen, Partner, Dependents, tenure
    - PhoneService, MultipleLines, InternetService
    - OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
    - StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod
    - MonthlyCharges, TotalCharges
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! {len(df)} customers found.")
            
            st.subheader("Preview of uploaded data")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🔮 Predict Churn for All Customers", type="primary"):
                with st.spinner("Analyzing customers..."):
                    predictions = []
                    
                    for idx, row in df.iterrows():
                        customer_dict = row.to_dict()
                        prob = predict_churn(customer_dict)
                        
                        if prob < 0.3:
                            risk = "Low Risk"
                        elif prob < 0.6:
                            risk = "Medium Risk"
                        else:
                            risk = "High Risk"
                        
                        predictions.append({
                            'Customer_ID': customer_dict.get('customerID', f'CUST-{idx}'),
                            'Churn_Probability': f"{prob:.2%}",
                            'Risk_Band': risk,
                            'Early_Warning': "YES" if prob >= 0.5 else "NO"
                        })
                    
                    results_df = pd.DataFrame(predictions)
                    
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Customers", len(results_df))
                    with col2:
                        high_risk = len(results_df[results_df['Risk_Band'] == 'High Risk'])
                        st.metric("High Risk", high_risk)
                    with col3:
                        medium_risk = len(results_df[results_df['Risk_Band'] == 'Medium Risk'])
                        st.metric("Medium Risk", medium_risk)
                    with col4:
                        low_risk = len(results_df[results_df['Risk_Band'] == 'Low Risk'])
                        st.metric("Low Risk", low_risk)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ============================================
# PAGE 3: ABOUT
# ============================================

elif page == "ℹ️ About":
    st.header("About This System")
    
    st.markdown("""
    ## 🎯 Purpose
    This system predicts customer churn probability and provides actionable retention recommendations.
    
    ## 🧠 How It Works
    1. **Data Collection**: Customer demographic, service, and billing information
    2. **Feature Engineering**: Creates risk indicators from raw data
    3. **ML Prediction**: Logistic Regression model predicts churn probability
    4. **Counterfactual Analysis**: Simulates different retention strategies
    5. **Cost-Benefit Analysis**: Ranks recommendations by ROI
    
    ## 📊 Risk Bands
    - **Low Risk (0-30%)**: Satisfied customers, low churn probability
    - **Medium Risk (30-60%)**: Monitor closely, consider preventive actions
    - **High Risk (60-100%)**: Immediate intervention required
    
    ## 💡 Key Features
    - Real-time churn prediction
    - Cost-aware retention recommendations
    - Batch processing for multiple customers
    - Interactive dashboard
    - REST API for integration
    
    ## 🛠️ Technology Stack
    - **ML Framework**: Scikit-learn
    - **Web Framework**: Streamlit / Flask
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly
    
    ## 📧 Contact
    For questions or support, contact your data science team.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | © 2024 Churn Prediction System")