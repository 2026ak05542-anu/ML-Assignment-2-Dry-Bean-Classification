import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

# Application title
st.title("🌱 Dry Bean Classification")
st.write(
    "Multi-class classification of Dry Bean varieties "
    "using five machine learning algorithms."
)

st.markdown("---")

# Model file paths
MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Gaussian Naive Bayes": "models/gaussian_naive_bayes.joblib",
    "SVM": "models/svm.joblib"
}

# Expected input features
FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

TARGET = "Class"

# Upload test dataset
st.subheader("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing all 16 features and the Class column",
    type=["csv"]
)

# Model selection
st.subheader("2. Select Classification Model")

selected_model = st.selectbox(
    "Choose a machine learning model",
    list(MODEL_FILES.keys())
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("3. Uploaded Dataset Preview")
        st.dataframe(data.head())

        # Validate columns
        required_columns = FEATURES + [TARGET]
        missing_columns = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
            st.stop()

        # Separate features and target
        X = data[FEATURES].copy()
        y = data[TARGET]

        # Check numeric features
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isnull().any().any() or y.isnull().any():
            st.error(
                "The uploaded CSV contains missing or invalid values."
            )
            st.stop()

        # Load saved model
        model_path = MODEL_FILES[selected_model]

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = joblib.load(model_path)

        # Predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)

        # Evaluation metrics
        accuracy = accuracy_score(y, y_pred)

        try:
            auc = roc_auc_score(
                y,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=model.classes_
            )
        except ValueError:
            auc = None

        precision = precision_score(
            y, y_pred,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y, y_pred,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y, y_pred,
            average="weighted",
            zero_division=0
        )

        mcc = matthews_corrcoef(y, y_pred)

        # Display evaluation results
        st.markdown("---")
        st.subheader("4. Model Evaluation Results")

        st.write(f"**Selected Model:** {selected_model}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric(
            "AUC",
            f"{auc:.4f}" if auc is not None else "Unavailable"
        )
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        st.markdown("---")
        st.subheader("5. Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred,
            labels=model.classes_
        )

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax
        )

        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        ax.set_title("Confusion Matrix")

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # Classification report
        st.markdown("---")
        st.subheader("6. Classification Report")

        report = classification_report(
            y,
            y_pred,
            zero_division=0
        )

        st.text(report)

        st.success("Model evaluation completed successfully!")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info(
        "Please upload a test CSV file to evaluate the selected model."
    )
