"""
SAP-RPT-1-OSS Predictor Demo App
A Streamlit UI to test the SAP tabular foundation model locally.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Page config
st.set_page_config(
    page_title="SAP-RPT-1-OSS Predictor",
    page_icon="🔮",
    layout="wide"
)

# Header
st.title("🔮 SAP-RPT-1-OSS Predictor Demo")
st.markdown("""
Test SAP's open source tabular foundation model for predictive analytics.
Upload your data, select training examples, and get predictions!
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Check if model is available
USE_MOCK = st.sidebar.checkbox("Use Mock Predictions (no GPU required)", value=True)

if not USE_MOCK:
    st.sidebar.warning("⚠️ Real model requires GPU and HuggingFace authentication")
    
prediction_type = st.sidebar.selectbox(
    "Prediction Type",
    ["Classification", "Regression"]
)

# Model parameters
st.sidebar.subheader("Model Parameters")
max_context_size = st.sidebar.slider("Max Context Size", 512, 8192, 4096, 512)
bagging = st.sidebar.slider("Bagging Iterations", 1, 10, 4)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Resources
- [SAP-RPT-1-OSS GitHub](https://github.com/SAP-samples/sap-rpt-1-oss)
- [Model on HuggingFace](https://huggingface.co/SAP/sap-rpt-1-oss)
- [Skill Repository](https://github.com/amitlals/sap-rpt1-oss-predictor)
""")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Training Data")
    
    # Sample data option
    use_sample = st.checkbox("Use sample data", value=True)
    
    if use_sample:
        sample_type = st.selectbox(
            "Select sample dataset",
            ["Customer Churn", "Payment Default"]
        )
        
        # Load sample data
        examples_path = Path(__file__).parent.parent / "examples"
        
        if sample_type == "Customer Churn":
            sample_file = examples_path / "customer_churn_sample.csv"
        else:
            sample_file = examples_path / "payment_default_sample.csv"
        
        if sample_file.exists():
            train_df = pd.read_csv(sample_file)
            st.success(f"✅ Loaded {len(train_df)} training examples")
            st.dataframe(train_df.head(10), use_container_width=True)
        else:
            st.error(f"Sample file not found: {sample_file}")
            train_df = None
    else:
        uploaded_train = st.file_uploader("Upload Training CSV", type="csv", key="train")
        if uploaded_train:
            train_df = pd.read_csv(uploaded_train)
            st.success(f"✅ Loaded {len(train_df)} training examples")
            st.dataframe(train_df.head(10), use_container_width=True)
        else:
            train_df = None

with col2:
    st.header("🎯 Test Data")
    
    uploaded_test = st.file_uploader("Upload Test CSV (without target column)", type="csv", key="test")
    
    if uploaded_test:
        test_df = pd.read_csv(uploaded_test)
        st.success(f"✅ Loaded {len(test_df)} test examples")
        st.dataframe(test_df.head(10), use_container_width=True)
    elif train_df is not None:
        st.info("💡 Or create test data from training data below")
        if st.button("Generate Test Sample (5 random rows)"):
            test_df = train_df.sample(min(5, len(train_df))).copy()
            st.session_state['test_df'] = test_df
    
    if 'test_df' in st.session_state:
        test_df = st.session_state['test_df']
        st.dataframe(test_df, use_container_width=True)
    elif not uploaded_test:
        test_df = None

# Target column selection
st.markdown("---")
st.header("🎯 Configure Prediction")

if train_df is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        target_column = st.selectbox(
            "Select Target Column (what to predict)",
            train_df.columns.tolist()
        )
    
    with col2:
        feature_columns = st.multiselect(
            "Select Feature Columns (or leave empty for all)",
            [c for c in train_df.columns if c != target_column],
            default=[]
        )
        if not feature_columns:
            feature_columns = [c for c in train_df.columns if c != target_column]

# Prediction button
st.markdown("---")

if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
    if train_df is None:
        st.error("Please upload or select training data first!")
    elif test_df is None and 'test_df' not in st.session_state:
        st.error("Please upload test data or generate a sample!")
    else:
        if 'test_df' in st.session_state and test_df is None:
            test_df = st.session_state['test_df']
        
        with st.spinner("Running prediction..."):
            
            # Prepare data
            X_train = train_df[feature_columns]
            y_train = train_df[target_column]
            X_test = test_df[feature_columns] if all(c in test_df.columns for c in feature_columns) else test_df
            
            if USE_MOCK:
                # Mock predictions for demo
                import time
                time.sleep(1)  # Simulate processing
                
                if prediction_type == "Classification":
                    unique_classes = y_train.unique()
                    predictions = np.random.choice(unique_classes, size=len(X_test))
                    confidence = np.random.uniform(0.7, 0.99, size=len(X_test))
                else:
                    mean_val = y_train.mean()
                    std_val = y_train.std()
                    predictions = np.random.normal(mean_val, std_val * 0.3, size=len(X_test))
                    confidence = None
                
                st.success("✅ Mock predictions complete!")
                
            else:
                # Real model
                try:
                    from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
                    
                    if prediction_type == "Classification":
                        model = SAP_RPT_OSS_Classifier(
                            max_context_size=max_context_size,
                            bagging=bagging
                        )
                    else:
                        model = SAP_RPT_OSS_Regressor(
                            max_context_size=max_context_size,
                            bagging=bagging
                        )
                    
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    confidence = None
                    
                    st.success("✅ Real model predictions complete!")
                    
                except ImportError:
                    st.error("""
                    ❌ SAP-RPT-1-OSS model not installed!
                    
                    Install with:
                    ```bash
                    pip install git+https://github.com/SAP-samples/sap-rpt-1-oss
                    huggingface-cli login
                    ```
                    """)
                    predictions = None
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    predictions = None
            
            # Display results
            if predictions is not None:
                st.header("📈 Prediction Results")
                
                results_df = X_test.copy()
                results_df[f"Predicted_{target_column}"] = predictions
                if confidence is not None:
                    results_df["Confidence"] = [f"{c:.1%}" for c in confidence]
                
                st.dataframe(results_df, use_container_width=True)
                
                # Download button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # Statistics
                st.subheader("📊 Prediction Statistics")
                
                if prediction_type == "Classification":
                    col1, col2, col3 = st.columns(3)
                    unique, counts = np.unique(predictions, return_counts=True)
                    
                    with col1:
                        st.metric("Total Predictions", len(predictions))
                    with col2:
                        st.metric("Unique Classes", len(unique))
                    with col3:
                        if confidence is not None:
                            st.metric("Avg Confidence", f"{np.mean(confidence):.1%}")
                    
                    # Class distribution
                    st.bar_chart(pd.Series(predictions).value_counts())
                    
                else:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Count", len(predictions))
                    with col2:
                        st.metric("Mean", f"{np.mean(predictions):.2f}")
                    with col3:
                        st.metric("Min", f"{np.min(predictions):.2f}")
                    with col4:
                        st.metric("Max", f"{np.max(predictions):.2f}")
                    
                    # Distribution
                    st.line_chart(predictions)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    SAP-RPT-1-OSS Predictor Skill | 
    <a href="https://github.com/amitlals/sap-rpt1-oss-predictor">GitHub</a> | 
    Apache 2.0 License
</div>
""", unsafe_allow_html=True)
