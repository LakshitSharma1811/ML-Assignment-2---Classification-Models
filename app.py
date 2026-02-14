import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report

# Set page configuration
st.set_page_config(page_title="ML Assignment 2 - Classification Models", layout="wide")

st.title("ML Assignment 2: Classification Model Comparison")
st.markdown("Implemented by [Lakshit Sharma/2025aa05485]")

# --- Step 1: Dataset Loading ---
st.sidebar.header("1. Data Loading")
data_source = st.sidebar.radio("Choose Data Source", ["Use Default (Breast Cancer)", "Upload CSV"])

X, y = None, None
df = None

if data_source == "Use Default (Breast Cancer)":
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    st.write("### Dataset: Breast Cancer Wisconsin (Diagnostic)")
    st.write(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    
elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV (Target column must be named 'target')", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'target' in df.columns:
            X = df.drop('target', axis=1)
            y = df['target']
            st.write("### User Uploaded Dataset")
            st.write(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        else:
            st.error("Uploaded CSV must contain a column named 'target'.")

# --- Step 2: Preprocessing ---
if X is not None and y is not None:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Step 3: Model Selection ---
    st.sidebar.header("2. Model Selection")
    model_name = st.sidebar.selectbox("Choose a Classification Model", 
        ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
    )

    model = None
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Train Model
    if st.sidebar.button("Train & Evaluate"):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        # --- Step 4: Evaluation Metrics ---
        st.header(f"Results for {model_name}")
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("MCC Score", f"{mcc:.4f}")
        col6.metric("AUC Score", f"{auc if isinstance(auc, str) else f'{auc:.4f}'}")

        # --- Step 5: Confusion Matrix & Report ---
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload a CSV or select the default dataset to proceed.")
