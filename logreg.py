import numpy as np

class LogisticRegression:
    def __init__ (self, lr = .01, epochs = 500, num_classes = 5):
        self.lr = lr
        self.epochs = epochs
        self.num_classes = 5
        self.weights = None
        self.bias = None
    
    def softmax(self, z):
        Z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_scores = np.exp(Z_shifted)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def one_hot_encode(self, y):
        y = np.array(y, dtype=int)
        y = y-1
        m = y.shape[0]
        Y_encoded = np.zeros((m, self.num_classes))
        for i in range(m):
             Y_encoded[i, y[i]] = 1
        return Y_encoded
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))
        Y_encoded = self.one_hot_encode(y)

        #one hot encode labels
        y_one_hot = self.one_hot_encode(y)

        for _ in range(self.epochs):
            #linear prediction
            logits = np.dot(X, self.weights) + self.bias
            probs = self.softmax(logits) #convert to probabilites
            

            #compute gradients
            dz = (probs - Y_encoded)
            dw = (1/n_samples) * np.dot(X.T, dz)
            db = (1/n_samples) * np.sum(dz, axis=0, keepdims=True)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict_proba(self, X):
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)
    
    def predict(self, X):
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)

    

