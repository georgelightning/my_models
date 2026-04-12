import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification # For generating synthetic classification data

class LogisticRegression:

    def __init__(self, learning_rate=0.01, n_iterations=1000):

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def add_bias(self, X):
        """Adds a column of 1s to the data"""
        return np.insert(X, 0, 1, axis=1)

    def _sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        """Fits the logistic regression model to the training data using Gradient Descent.

        Args:
            X (np.ndarray): The input features (training data).
                            Expected shape: (n_samples, n_features).
            y (np.ndarray): The target labels (0 or 1).
                            Expected shape: (n_samples,)."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        X_b = self.add_bias(X)
        n_of_samples, n_of_features = X_b.shape
        self.weights = np.zeros((n_of_features, 1))

        print(f"Fitting using Gradient Descent for {self.n_iterations} iterations...")

        for i in range(self.n_iterations):
            z = X_b @ self.weights
            pred = self._sigmoid(z)
            error = pred - y
            grad = (1/n_of_samples) * (X_b.T @ error)
            self.weights -= self.learning_rate * grad
            epsilon = 1e-10
            cost = -(1/n_of_samples) * (y.T @ np.log(pred + epsilon) + (-y + 1).T @ np.log(-pred + 1 + epsilon)).item()
            self.cost_history.append(cost)

            if (i + 1) % (self.n_iterations / 10) == 0:
                print(f"Iteration {i + 1}/{self.n_iterations}, Cost: {cost:.4f}")
        print("Gradient Descent fitting complete.")
        self.bias = self.weights[0, 0]
        self.weights = self.weights[1:]  # Feature weights
    def predict_probabilities(self, X):
        """
                Predicts probabilities for the positive class (class 1).
                Args:
                    X (np.ndarray): The input features for prediction.
                                    Expected shape: (n_samples, n_features).
                Returns:
                    np.ndarray: Predicted probabilities for class 1.
                                Shape: (n_samples,).
                """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
        X_b = self.add_bias(X)
        full_weights = np.insert(self.weights, 0, self.bias).reshape(-1, 1)
        z = X_b @ full_weights
        return self._sigmoid(z).flatten()
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_probabilities(X)
        return (probabilities >= threshold).astype(int)

    def calculate_accuracy(self, y_true, y_pred):
        """
        Calculates the accuracy score.
        Args:
            y_true (np.ndarray): The true labels.
            y_pred (np.ndarray): The predicted labels.
        Returns:
            float: The accuracy score (proportion of correct predictions).
        """
        return np.mean(y_true.flatten() == y_pred.flatten())







