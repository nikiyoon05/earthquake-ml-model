import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.decomposition import PCA
from data_preprocessing import load_process_data

def compute_svd(X_train, X_test, n_components=20):
    """
    Apply Singular Value Decomposition (SVD) to reduce dimensionality.

    Parameters:
    - X_train: Training feature matrix
    - X_test: Test feature matrix
    - n_components: Number of components to keep

    Returns:
    - X_train_k: Reduced training features
    - X_test_k: Reduced test features
    """
    U, S, Vt = svd(X_train, full_matrices=False)
    U_test, S_test, Vt_test = svd(X_test, full_matrices=False)

    def plot_variance(U, S, Vt):
        """
        Plot cumulative variance explained by SVD components.
        """
        total_variance = np.sum(S**2)
        variance_explained = np.cumsum(S**2) / total_variance
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(S) + 1), variance_explained, marker='o', linestyle='--', color='b')
        plt.xlabel("Rank k (Number of Components)")
        plt.ylabel("Proportional Variance Explained")
        plt.title("Variance Explained vs. Rank k")
        plt.axhline(y=0.95, color='r', linestyle='--', label="95% Variance Explained")
        plt.legend()
        plt.grid()
        plt.show()
    
    # Reduce dimensionality using top k components
    X_train_k = U[:, :n_components] @ np.diag(S[:n_components]) @ Vt[:n_components]
    X_test_k = U_test[:, :n_components] @ np.diag(S_test[:n_components]) @ Vt_test[:n_components]

    plot_variance(U, S, Vt)

    return X_train_k, X_test_k


def compute_pca(X_train, X_test, n_components=20):
    """
    Apply Principal Component Analysis (PCA) and visualize feature importance.

    Parameters:
    - X_train: Training feature matrix (Pandas DataFrame)
    - X_test: Test feature matrix (Pandas DataFrame)
    - n_components: Number of PCA components to keep

    Returns:
    - X_train_pca: Reduced training features
    - X_test_pca: Reduced test features
    - pca: Trained PCA object
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    feature_names = X_train.columns  # assumes DataFrame input

    # Print top 5 contributing features per principal component
    for i in range(pca.n_components_):
        loadings = pca.components_[i]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        top_features = feature_names[sorted_idx[:5]]
        print(f"PC{i+1} top 5 features:")
        for feat in top_features:
            print("  ", feat, loadings[feature_names.get_loc(feat)])

    def plot_biplot(pca, X_df):
        """
        Plot data in PC1–PC2 space with reference axes.
        """
        n_plot = min(400, X_train_pca.shape[0])
        subset_idx = np.random.choice(X_train_pca.shape[0], n_plot, replace=False)
        X_sub = X_train_pca[subset_idx, :]

        plt.figure(figsize=(8, 6))
        plt.scatter(X_sub[:, 0], X_sub[:, 1], alpha=0.6, label="Data Subset")

        # Add PC1 and PC2 reference lines
        x_min, x_max = X_train_pca[:, 0].min(), X_train_pca[:, 0].max()
        y_min, y_max = X_train_pca[:, 1].min(), X_train_pca[:, 1].max()
        plt.plot([x_min, x_max], [0, 0], color='red', linewidth=1.5)
        plt.plot([0, 0], [y_min, y_max], color='green', linewidth=1.5)
        plt.text(x_max * 0.95, 0.02 * (y_max - y_min), "PC1", color='red', fontsize=10)
        plt.text(0.02 * (x_max - x_min), y_max * 0.95, "PC2", color='green', fontsize=10)

        plt.title("Data in PC1–PC2 space")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        plt.legend()
        plt.show()

    plot_biplot(pca, X_train)

    return X_train_pca, X_test_pca, pca


if __name__ == "__main__":
    # Example run (will execute if run as standalone script)
    X_train, X_test, _, _ = load_process_data("csv_building_structure.csv")
    X_train_svd, X_test_svd = compute_svd(X_train, X_test)
    X_train_pca, X_test_pca, pca = compute_pca(X_train, X_test)
