import numpy as np
import matplotlib.pyplot as plt
import requests
import io # To treat the downloaded string as a file
import pandas as pd
import time


class TwoLayerNN:

    def __init__(self, n_x, n_h, n_y, learning_rate: int=0.01):
        """
        :param n_x: the size of the input vector
        :param n_y: the size of the output layer
        :param n_h: the size of the hidden layer
        """
        self.n_x = n_x
        self.n_y = n_y
        self.n_h = n_h
        self.learning_rate = learning_rate
        self.weights = self._initialize_parameters()
        self.cache = {}

    def _initialize_parameters(self):
        w1 = np.random.randn(self.n_h, self.n_x)
        w2 = np.random.randn(self.n_y, self.n_h)
        b1 = np.zeros((self.n_h, 1))
        b2 = np.zeros((self.n_y, 1))
        return {"W1": w1, "W2": w2, "b1": b1, "b2": b2}
    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=0)

    @staticmethod
    def relu(Z):
        A = np.maximum(0, Z)
        return A

    def forward_pass(self, X):
        A = X
        self.cache = {}
        for i in range(1, 3):
            # 1. Linear Combination: Z[i] = W[i] @ A[i-1] + b[i]
            Z = self.weights[f"W{i}"] @ A + self.weights[f"b{i}"]
            self.cache[f"Z{i}"] = Z

            # 2. Activation Function
            if i == 1:
                A = TwoLayerNN.relu(Z)

            elif i == 2:
                A = TwoLayerNN.softmax(Z)


            self.cache[f"A{i}"] = A
        return A

    @staticmethod
    def cost(A2, Y):
        """
            Computes the Categorical Cross-Entropy Loss.
            A2: The Softmax output (A[2]) 3 by m
            Y: The One-Hot encoded true labels 3 by m
        """

        m = Y.shape[1]
        log_A2 = np.log(A2 + 1e-10) #to prevent log(0)
        cost_matrix = Y * log_A2
        cost = -(1 / m) * np.sum(cost_matrix)

        return np.squeeze(cost)

    def backpropogation(self, X, Y):
        """
                Calculates the gradients (dW and db) using self.cache.

                Note: The X and Y are passed here, as they change with mini-batching,
                      but the intermediate values are in self.cache.
        """
        m = X.shape[1]
        # Retrieve values from the instance cache
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]
        Z1 = self.cache["Z1"]

        # Retrieve W2 from instance parameters
        W2 = self.weights["W2"]
        grads = {}
        #Gradients
        dZ2 = A2 - Y
        dW2 = (1 / m) * (dZ2 @ A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * np.int64(Z1 > 0) #np.int64(Z1 > 0) is just the derivative of ReLu
        dW1 = (1/m) * (dZ1 @ X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def fit(self, X, y, epochs=10000):
        """
                Runs the main training loop using Gradient Descent.

                Arguments:
                X              -- Input data (features, samples)
                Y              -- True labels (output classes, samples)
                num_iterations -- Number of epochs to train for
                """
        costs = []

        for i in range(0, epochs):

            # 1. Forward Propagation (Prediction and Caching)
            # A2 is the final prediction; self.cache is updated internally
            A2 = self.forward_pass(X)

            # 2. Compute Cost (Loss)
            cost = self.cost(A2, y)

            # grads contains dW and db
            grads = self.backpropogation(X, y)

            self.update_parameters(grads)

            if i % 1000 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost:.4f}")

        return self.weights, costs

    def update_parameters(self, grads):
        """
        Updates the model's parameters using the Gradient Descent update rule:
        Parameter = Parameter - learning_rate * Gradient

        Arguments:
        grads -- dictionary containing dW1, db1, dW2, db2 (the gradients)
        """

        # The update uses the learning_rate attribute stored in the instance
        alpha = self.learning_rate

        # Update Weights and Biases for Layer 1
        self.weights["W1"] -= alpha * grads["dW1"]
        self.weights["b1"] -= alpha * grads["db1"]

        # Update Weights and Biases for Layer 2
        self.weights["W2"] -= alpha * grads["dW2"]
        self.weights["b2"] -= alpha * grads["db2"]

        # Note: Since 'self.parameters' is modified directly,
        # there is no explicit 'return self.parameters' needed.

    def predict_accuracy(self, X, y):
        """
        Calculates the accuracy of the model predictions.

        Arguments:
        X -- Data matrix (features, samples)
        Y -- True labels (one-hot encoded: output classes, samples)

        Returns:
        accuracy (float)
        """
        A2 = self.forward_pass(X)

        predictions = np.argmax(A2, axis=0)

        true_labels = np.argmax(y, axis=0)

        accuracy = np.mean(predictions == true_labels)

        return accuracy

