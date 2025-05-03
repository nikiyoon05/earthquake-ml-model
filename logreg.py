import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=500, num_classes=5):
        """
        Initialize the logistic regression model.

        Parameters:
        - lr: Learning rate for gradient descent
        - epochs: Number of training iterations
        - num_classes: Number of target classes
        """
        self.lr = lr
        self.epochs = epochs
        self.num_classes = num_classes
        self.weights = None
        self.bias = None

    def softmax(self, z):
        """
        Compute softmax probabilities for input z.

        Parameters:
        - z: Linear logits, shape (n_samples, num_classes)

        Returns:
        - Softmax probabilities
        """
        Z_shifted = z - np.max(z, axis=1, keepdims=True)  # For numerical stability
        exp_scores = np.exp(Z_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def one_hot_encode(self, y):
        """
        One-hot encode the class labels.

        Parameters:
        - y: Array of class labels, shape (n_samples,)

        Returns:
        - One-hot encoded matrix, shape (n_samples, num_classes)
        """
        y = np.array(y, dtype=int) - 1  # Convert labels to 0-indexed
        m = y.shape[0]
        Y_encoded = np.zeros((m, self.num_classes))
        for i in range(m):
            Y_encoded[i, y[i]] = 1
        return Y_encoded

    def fit(self, X, y):
        """
        Train the logistic regression model using softmax and cross-entropy.

        Parameters:
        - X: Feature matrix, shape (n_samples, n_features)
        - y: Target labels, shape (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))
        Y_encoded = self.one_hot_encode(y)

        for _ in range(self.epochs):
            # Compute logits and apply softmax
            logits = np.dot(X, self.weights) + self.bias
            probs = self.softmax(logits)

            # Compute gradients
            dz = probs - Y_encoded
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz, axis=0, keepdims=True)

            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        """
        Predict class probabilities for given input.

        Parameters:
        - X: Feature matrix, shape (n_samples, n_features)

        Returns:
        - Probability matrix, shape (n_samples, num_classes)
        """
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)

    def predict(self, X):
        """
        Predict class labels for given input.

        Parameters:
        - X: Feature matrix, shape (n_samples, n_features)

        Returns:
        - Predicted class labels, shape (n_samples,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
