import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # Making the predictions
            y_pred = np.dot(self.weights, X) + self.bias

            # Calculating the gradients
            dw = (1/n_samples) * (2 * np.dot(X.T, (y_pred - y)))
            db = (1/n_samples) * (2 * np.sum(y_pred - y))

            # Updating the weights & biases
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_pred = np.dot(self.weights, X) + self.bias
        return y_pred

if __name__ == "__main__":
    pass