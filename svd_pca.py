import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from scipy.linalg import svd
import matplotlib.pyplot as plt
from data_preprocessing import load_process_data

def compute_svd(X_train, X_test, n_components=20):
    U, S, Vt = svd(X_train, full_matrices=False)
    U_test, S_test, Vt_test = svd(X_test, full_matrices=False)


    def plot_variance(U, S, Vt):
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
        

    X_train_k = U[:, :n_components] @ np.diag(S[:n_components]) @ Vt[:n_components]
    X_test_k = U_test[:, :n_components] @ np.diag(S_test[:n_components]) @ Vt_test[:n_components]
    
    X_train = U
    X_test = U_test
    plot_variance(U, S, Vt)

    return X_train_k, X_test_k


def compute_pca(X_train, X_test, n_components=20):
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    feature_names = X_train.columns  # or keep track of them if X_train is a DataFrame

    #top features in each component
    for i in range(pca.n_components_):
        loadings = pca.components_[i]        # shape: (n_features,)
        sorted_idx = np.argsort(np.abs(loadings))[::-1]  # sort by absolute weight
        top_features = feature_names[sorted_idx[:5]]     # top 5 features
        print(f"PC{i+1} top 5 features:")
        for feat in top_features:
            print("  ", feat, loadings[feature_names.get_loc(feat)])
    def plot_biplot(pca, X_df):
        n_plot = min(400, X_train_pca.shape[0])  # up to 300 points
        subset_idx = np.random.choice(X_train_pca.shape[0], n_plot, replace=False)
        X_sub = X_train_pca[subset_idx, :]

        #3. Plot the data
        plt.figure(figsize=(8,6))
        plt.scatter(X_sub[:,0], X_sub[:,1], alpha=0.6, label="Data Subset")

        #4. Draw lines for PC1 and PC2 (simple approach)
        #pick a scale factor so the lines appear within the data range
        x_min, x_max = X_train_pca[:,0].min(), X_train_pca[:,0].max()
        y_min, y_max = X_train_pca[:,1].min(), X_train_pca[:,1].max()
        
        #Let's define endpoints for lines:
        #PC1 is a horizontal line from (x_min, 0) to (x_max, 0)
        #PC2 is a vertical line from (0, y_min) to (0, y_max)
        plt.plot([x_min, x_max], [0, 0], color='red', linewidth=1.5)
        plt.plot([0, 0], [y_min, y_max], color='green', linewidth=1.5)
        
        #Label them "PC1" and "PC2" near the right/top ends
        plt.text(x_max*0.95, 0.02*(y_max-y_min), "PC1", color='red', fontsize=10)
        plt.text(0.02*(x_max-x_min), y_max*0.95, "PC2", color='green', fontsize=10)

        plt.title("Data in PC1–PC2 space")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
        plt.legend()
        plt.show()
    plot_biplot(pca, X_train)
    

    return X_train_pca, X_test_pca, pca

if __name__ == "__main__":

    X_train, X_test, _, _ = load_process_data("csv_building_structure.csv")
    X_train_svd, X_test_svd = compute_svd(X_train, X_test)
    X_train_pca, X_test_pca, pca = compute_pca(X_train, X_test)
        
 


