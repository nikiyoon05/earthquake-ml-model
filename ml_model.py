import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import load_process_data
from sklearn.linear_model import LogisticRegression
from svd_pca import compute_svd, compute_pca
from logreg import LogisticRegression as LR
from sklearn.metrics import f1_score

def train_random_forest(X_train, X_test, y_train, y_test):
    #train random foest

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    return accuracy, report

if __name__ == "__main__":
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_process_data("csv_building_structure.csv")
    
    # Compute SVD and PCA components
    X_train_svd, X_test_svd = compute_svd(X_train, X_test)
    X_train_pca, X_test_pca, _ = compute_pca(X_train, X_test)

    #logistic regression with svd
    model = LogisticRegression()
    model.fit(X_train_svd, y_train)
    preds = model.predict(X_test_svd)
    print(accuracy_score(y_test, preds))
    print(f1_score(y_test, preds, average="macro"))

    #logistic regression with pca
    model = LogisticRegression()
    model.fit(X_train_pca, y_train)
    preds2 = model.predict(X_test_pca)
    print(accuracy_score(y_test, preds2))
    print(f1_score(y_test, preds2, average="macro"))

    #from scratch logistic regressoin with svd
    logregScratch = LR()
    logregScratch.fit(X_train_svd, y_train)
    predictions = logregScratch.predict(X_test_svd)
    print(accuracy_score(y_test, predictions))

    #knn model with svd
    neigh = KNeighborsClassifier()
    neigh.fit(X_train_svd, y_train)
    y_pred_kn = neigh.predict(X_test_svd)
    score = accuracy_score(y_test, y_pred_kn)
    print(f"Accuracy score for KNN: {score}")
    
    # Train Random Forest on SVD components
    acc_svd, report_svd = train_random_forest(X_train_svd, X_test_svd, y_train, y_test)
    print(f"Random Forest Accuracy with SVD Components: {acc_svd:.4f}")
    print("Classification Report for SVD:")
    print(report_svd)
    
    # Train Random Forest on PCA components
    acc_pca, report_pca = train_random_forest(X_train_pca, X_test_pca, y_train, y_test)
    print(f"Random Forest Accuracy with PCA Components: {acc_pca:.4f}")
    print("Classification Report for PCA:")
    print(report_pca)
    
    # Compare results
    print("\n=== Model Comparison ===")
    print(f"Accuracy (SVD): {acc_svd:.4f}")
    print(f"Accuracy (PCA): {acc_pca:.4f}")
    if acc_svd > acc_pca:
        print("SVD performed better!")
    else:
        print("PCA performed better!")
