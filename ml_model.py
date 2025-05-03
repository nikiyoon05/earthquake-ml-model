import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression

from data_preprocessing import load_process_data
from svd_pca import compute_svd, compute_pca
from logreg import LogisticRegression as LR  # Custom implementation

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train a Random Forest classifier and return accuracy and classification report.

    Parameters:
    - X_train, X_test: Feature matrices for training and testing
    - y_train, y_test: Corresponding labels

    Returns:
    - accuracy: Accuracy score on the test set
    - report: Full classification report as string
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

if __name__ == "__main__":
    # Load and preprocess the dataset
    X_train, X_test, y_train, y_test = load_process_data("csv_building_structure.csv")
    
    # Compute dimensionality-reduced features
    X_train_svd, X_test_svd = compute_svd(X_train, X_test)
    X_train_pca, X_test_pca, _ = compute_pca(X_train, X_test)

    # Logistic Regression using SVD features (sklearn)
    model = LogisticRegression()
    model.fit(X_train_svd, y_train)
    preds = model.predict(X_test_svd)
    print("Logistic Regression with SVD:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score (macro):", f1_score(y_test, preds, average="macro"))

    # Logistic Regression using PCA features (sklearn)
    model = LogisticRegression()
    model.fit(X_train_pca, y_train)
    preds2 = model.predict(X_test_pca)
    print("\nLogistic Regression with PCA:")
    print("Accuracy:", accuracy_score(y_test, preds2))
    print("F1 Score (macro):", f1_score(y_test, preds2, average="macro"))

    # Logistic Regression from scratch (custom) using SVD
    logregScratch = LR()
    logregScratch.fit(X_train_svd, y_train)
    predictions = logregScratch.predict(X_test_svd)
    print("\nCustom Logistic Regression with SVD:")
    print("Accuracy:", accuracy_score(y_test, predictions))

    # K-Nearest Neighbors with SVD features
    neigh = KNeighborsClassifier()
    neigh.fit(X_train_svd, y_train)
    y_pred_kn = neigh.predict(X_test_svd)
    score = accuracy_score(y_test, y_pred_kn)
    print("\nK-Nearest Neighbors with SVD:")
    print(f"Accuracy score for KNN: {score}")

    # Random Forest with SVD features
    acc_svd, report_svd = train_random_forest(X_train_svd, X_test_svd, y_train, y_test)
    print("\nRandom Forest with SVD:")
    print(f"Accuracy: {acc_svd:.4f}")
    print("Classification Report:\n", report_svd)

    # Random Forest with PCA features
    acc_pca, report_pca = train_random_forest(X_train_pca, X_test_pca, y_train, y_test)
    print("\nRandom Forest with PCA:")
    print(f"Accuracy: {acc_pca:.4f}")
    print("Classification Report:\n", report_pca)

    # Compare PCA vs SVD performance
    print("\n=== Model Comparison ===")
    print(f"Accuracy (SVD): {acc_svd:.4f}")
    print(f"Accuracy (PCA): {acc_pca:.4f}")
    if acc_svd > acc_pca:
        print("SVD performed better!")
    else:
        print("PCA performed better!")
