import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

from policy_engine import PolicyEngine

# --- Configuration & Setup ---
st.set_page_config(page_title="MCP Security Guard Dashboard", layout="wide")

@st.cache_resource
def load_policy_engine():
    try:
        engine = PolicyEngine()
        return engine
    except Exception as e:
        st.error(f"Failed to load Policy Engine: {e}")
        return None

engine = load_policy_engine()

# --- Utility Functions ---
@st.cache_data
def load_logs():
    if os.path.exists("logs/security_log.csv"):
        return pd.read_csv("logs/security_log.csv")
    return pd.DataFrame()

@st.cache_data
def load_predictions():
    if os.path.exists("logs/model_predictions.csv"):
        return pd.read_csv("logs/model_predictions.csv")
    return pd.DataFrame()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
menu = ["Dashboard", "Tool Analyzer", "Security Logs", "Model Evaluation"]
choice = st.sidebar.selectbox("Go to", menu)

# --- Sections ---

if choice == "Dashboard" or choice == "Tool Analyzer":
    st.title("MCP Security Guard Dashboard")
    st.header("Tool Analyzer")
    st.markdown("Enter the description of an MCP tool below to evaluate its security risk.")
    
    tool_text = st.text_area("Tool Description", height=150, placeholder="E.g., run_shell execute bash command")
    
    if st.button("Evaluate Tool"):
        if tool_text.strip():
            if engine:
                with st.spinner("Analyzing..."):
                    # The actual function in policy_engine.py is evaluate_request
                    # and returns confidence_score instead of confidence
                    result = engine.evaluate_request(tool_text)
                    
                    risk_label = result.get('risk_label', 'Unknown')
                    confidence = result.get('confidence_score', 0.0)
                    decision = result.get('decision', 'Unknown')
                    
                    st.subheader("Results")
                    
                    # Color coding logic
                    if decision == "ALLOW":
                        st.success(f"**Decision:** {decision}")
                    elif decision == "WARN":
                        st.warning(f"**Decision:** {decision}")
                    else:
                        st.error(f"**Decision:** {decision}")
                        
                    col1, col2 = st.columns(2)
                    col1.metric("Risk Label", risk_label.upper())
                    col2.metric("Confidence Score", f"{confidence:.2%}")
                    
                    st.json(result)
            else:
                st.error("Policy Engine is not loaded.")
        else:
            st.warning("Please enter a tool description.")

if choice == "Dashboard" or choice == "Security Logs":
    if choice == "Security Logs":
        st.title("Security Logs Viewer")
    else:
        st.markdown("---")
        st.header("Security Logs Viewer")
        
    df_logs = load_logs()
    
    if not df_logs.empty:
        # Standardize columns to expected if possible
        display_cols = ['timestamp', 'tool_text', 'risk_label', 'confidence_score', 'decision']
        existing_cols = [c for c in display_cols if c in df_logs.columns]
        
        # rename confidence_score to confidence for display if desired, 
        # but let's just show existing columns matching criteria
        st.dataframe(df_logs[existing_cols] if existing_cols else df_logs, use_container_width=True)
    else:
        st.info("No security logs found at logs/security_log.csv")

if choice == "Dashboard" or choice == "Model Evaluation":
    if choice == "Model Evaluation":
        st.title("Model Evaluation Metrics")
    else:
        st.markdown("---")
        st.header("Model Evaluation")
        
    df_preds = load_predictions()
    
    if not df_preds.empty and 'true_label' in df_preds.columns and 'predicted_label' in df_preds.columns:
        st.subheader("Performance Metrics")
        
        y_true = df_preds['true_label']
        y_pred = df_preds['predicted_label']
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        # Using weighted average for multi-class
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision (Weighted)", f"{precision:.4f}")
        col3.metric("Recall (Weighted)", f"{recall:.4f}")
        col4.metric("F1 Score (Weighted)", f"{f1:.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        image_path = "logs/confusion_matrix.png"
        if os.path.exists(image_path):
            img = Image.open(image_path)
            st.image(img, caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix image not found at logs/confusion_matrix.png")
            
        # Risk Distribution
        st.subheader("Risk Distribution")
        
        # Count values from true labels or predictions
        # For evaluation, we show the distribution of the dataset
        label_counts = df_preds['true_label'].value_counts().reset_index()
        label_counts.columns = ['Risk Label', 'Count']
        
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=label_counts, x='Risk Label', y='Count', ax=ax, palette="viridis")
        ax.set_title("Distribution of True Labels in Evaluation Dataset")
        st.pyplot(fig)
        
    else:
        st.info("Model predictions not found or missing required columns at logs/model_predictions.csv")

