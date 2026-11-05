# Dry Bean Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate five supervised machine learning classification algorithms to identify dry bean varieties using their physical and geometrical characteristics.

The five machine learning algorithms implemented are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All five models are trained and evaluated using the same dataset and the same stratified train-test split. Their performance is compared using six evaluation metrics: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset Name:** Dry Bean Dataset

**Dataset Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

The Dry Bean Dataset contains measurements of dry bean samples belonging to seven different varieties: Seker, Barbunya, Bombay, Cali, Dermason, Horoz, and Sira.

The original dataset contains 13,611 instances and 16 numerical input features. After removing 68 duplicate records, the dataset used for this project contains 13,543 instances.

### Dataset Details

| Description | Details |
|---|---|
| Dataset Name | Dry Bean Dataset |
| Original Number of Instances | 13,611 |
| Instances After Removing Duplicates | 13,543 |
| Number of Input Features | 16 |
| Target Variable | Class |
| Number of Target Classes | 7 |
| Missing Values | None |

### Input Features

The dataset contains the following 16 numerical features:

1. Area
2. Perimeter
3. MajorAxisLength
4. MinorAxisLength
5. AspectRation
6. Eccentricity
7. ConvexArea
8. EquivDiameter
9. Extent
10. Solidity
11. roundness
12. Compactness
13. ShapeFactor1
14. ShapeFactor2
15. ShapeFactor3
16. ShapeFactor4

The target variable is `Class`, which identifies the dry bean variety.

### Train-Test Split

The dataset was divided into training and testing sets using an 80:20 stratified split with `random_state=42`.

| Dataset Split | Number of Instances |
|---|---:|
| Training Set | 10,834 |
| Testing Set | 2,709 |
| Total | 13,543 |

The same train-test split was used for all five classification algorithms to ensure a fair performance comparison.

## 3. GitHub Repository

**GitHub Repository Link:**

https://github.com/2026ak05542-anu/ML-Assignment-2-Dry-Bean-Classification

The repository contains the experiment notebook, saved trained machine learning models, test dataset, Streamlit application source code, and project dependencies.

## 4. Models Used and Performance Comparison

Five supervised machine learning classification algorithms were implemented and evaluated using the test dataset.

The evaluation metrics used are:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the ability of the model to distinguish between classes.
- **Precision:** Measures the correctness of positive class predictions.
- **Recall:** Measures the ability to identify actual class instances.
- **F1 Score:** Represents the harmonic mean of Precision and Recall.
- **MCC:** Measures the overall quality of classification predictions.

Precision, Recall, and F1 Score use weighted averaging. AUC uses weighted one-vs-rest multiclass evaluation.

### Model Performance Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.919158 | 0.993449 | 0.919726 | 0.919158 | 0.919290 | 0.902314 |
| Decision Tree | 0.896641 | 0.936251 | 0.896507 | 0.896641 | 0.896412 | 0.875028 |
| KNN | 0.915467 | 0.981051 | 0.916315 | 0.915467 | 0.915652 | 0.897791 |
| Gaussian Naive Bayes | 0.897010 | 0.989880 | 0.899653 | 0.897010 | 0.897199 | 0.876171 |
| SVM | 0.921004 | 0.993315 | 0.921411 | 0.921004 | 0.921078 | 0.904471 |

## 5. Model-wise Observations and Overall Winner

The following table summarizes the performance of each classification algorithm based on the evaluation results.

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved 91.92% accuracy and the highest AUC of 0.993449. It demonstrated strong overall classification performance and excellent class discrimination. |
| Decision Tree | Achieved 89.66% accuracy, which was comparatively lower than the other models. It provides an interpretable classification approach based on decision rules. |
| KNN | Achieved 91.55% accuracy and an MCC of 0.897791. It performed well, although its results were slightly below Logistic Regression and SVM. |
| Naive Bayes (Gaussian) | Achieved 89.70% accuracy and an AUC of 0.989880. It showed good class discrimination, but its overall classification performance was comparatively lower. |
| Support Vector Machine | Achieved the highest accuracy (92.10%), Precision, Recall, F1 Score, and MCC. It demonstrated the best overall performance across most evaluation metrics. |
| **Overall Winner for the Dataset** | **Support Vector Machine (SVM)** is the overall winner because it achieved the best results in five out of six evaluation metrics. Logistic Regression achieved the highest AUC, with a very small difference from SVM. |

### Overall Conclusion

Based on the experimental results, Support Vector Machine (SVM) performed best overall for the Dry Bean classification task. It achieved the highest Accuracy, Precision, Recall, F1 Score, and MCC among the five models.

Logistic Regression also performed very well and achieved the highest AUC of 0.993449.

Therefore, SVM is selected as the overall winning model based on its performance across the majority of evaluation metrics.

## 6. Streamlit Application

The Streamlit application provides an interactive interface for evaluating the trained dry bean classification models.

### Application Features

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- View Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- View the confusion matrix.
- View the classification report.

The application loads the previously trained and saved models from the repository. It evaluates the uploaded test dataset without retraining the models.

**Live Streamlit Application Link:**

https://dry-bean-classification-2026ak05542.streamlit.app/

