# Earthquake Damage Classification Using SVD and PCA

This project applies dimensionality reduction and machine learning techniques to classify structural damage from the 2015 Gorkha earthquake in Nepal. Using a dataset of over 762,000 buildings, we reduced data dimensionality with **Singular Value Decomposition (SVD)** and **Principal Component Analysis (PCA)**, then trained classifiers to predict damage severity on a 1–5 scale.

### 📊 Methods
- **Preprocessing:** One-hot encoding, normalization, and feature selection
- **Dimensionality Reduction:** SVD and PCA (reduced to 20 components)
- **Modeling:** Logistic Regression and Random Forest classifiers
- **Evaluation:** Accuracy and F1 score comparisons between models

### 📈 Results
| Model                 | Accuracy | F1 Score |
|----------------------|----------|----------|
| SVD + Logistic Reg.  | 86.4%    | 0.76     |
| PCA + Logistic Reg.  | 87.1%    | 0.85     |
| SVD + Random Forest  | 88.31%   | 0.88     |
| PCA + Random Forest  | 88.40%   | 0.88     |

 

For full project details, see the linked PDF>
[View Project Paper](Math_104_Project.pdf)

This project was completed as a final project for **Math 104: Applied Linear Algebra** at Stanford University.
