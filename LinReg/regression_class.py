import numpy as np
import matplotlib.pyplot as plt



class LinearRegression:
    """
    An implementation of linear regression, an honest try
    """

    def __init__(self, learning_rate = 0.01, n_of_iterations = 1000, method = "gradient_descent"):
        self.learning_rate = learning_rate
        self.n_of_iterations = n_of_iterations
        self.method = method #normal_equation or gradient_descent
        self.weights = None
        self.bias = None
        self.cost_history = list()

    def add_bias(self, X):
        return np.insert(X, 0, 1, axis=1)

    def fit(self, X, y):
        """
                Fits the linear regression model to the training data.

                Args:
                    X (np.ndarray): The input features (training data).
                                    Expected shape: (m_samples, n_features).
                    y (np.ndarray): The target values (training labels).
                                    Expected shape: (n_samples,).
        """
        print(y.shape)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        X_b = self.add_bias(X)
        num_of_parameters = X_b.shape[1]
        self.weights = np.zeros((num_of_parameters, 1))

        if self.method == "normal_equation":
            print("Fitting using normal equation...")
            # Calculating X_b_T * X_b
            X_T_X = X_b.T @ X_b
            try:
                inv = np.linalg.inv(X_T_X)
                self.weights = inv @ X_b.T @ y
                print("Normal equation fitting complete")
            except np.linalg.LinAlgError:
                print("The matrix is singular! Trying with gradient descent...")
                n = X_b.shape[0]
                for iteration in range(self.n_of_iterations):
                    prediction = X_b @ self.weights
                    error = prediction - y
                    grad = (1 / n) * X_b.T @ error
                    self.weights -= self.learning_rate * grad
                    cost = np.mean(error ** 2) / 2
                    self.cost_history.append(cost)
                    if (iteration % 10) == 0:
                        print(f"Epoch: {iteration}/{n}\n Cost: {cost}")
                print("Fitting with gradient descent complete.")

        elif self.method == "gradient_descent":
            n = X_b.shape[0]
            for iteration in range(self.n_of_iterations):
                prediction = X_b @ self.weights
                error = prediction - y
                grad = (1 / n) * X_b.T @ error
                self.weights -= self.learning_rate * grad
                cost = np.mean(error ** 2) / 2
                self.cost_history.append(cost)
                if (iteration % 10) == 0:
                    print(f"Epoch: {iteration}/{self.n_of_iterations}\n Cost: {cost}")
            print("Fitting with gradient descent complete.")
        else:
            raise ValueError("Wrong argument, valid arguments: \"gradient_descent\", \"normal_equation\"")

        if self.weights is not None:
            self.bias = self.weights[0, 0]
            self.weights = self.weights[1:]


    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted yet. Call .fit() first.")
        return (X @ self.weights + self.bias).flatten()

    def calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)




if __name__ == "__main__":
    # 1. Generate Synthetic Data
    np.random.seed(42) # for reproducibility
    n_samples = 100
    X_synthetic = 2 * np.random.rand(n_samples, 1) # Single feature
    y_synthetic = 4 + 3 * X_synthetic + np.random.randn(n_samples, 1) * 1.5 # y = 4 + 3x + noise



    n_samples = 1000
    n_features = 3
    np.random.seed(42)  # for reproducibility

    # Generate X from a normal distribution
    X = np.random.randn(n_samples, n_features)

    beta = np.array([2.5, -1.2, 0.7])  # True coefficients
    intercept = 3.0  # True intercept

    noise_std = 0.5
    noise = np.random.randn(n_samples) * noise_std

    y = intercept + X @ beta + noise

    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    mse_multi_gd = model.calculate_mse(y, preds)
    print(f"Multi-feature Gradient Descent - Bias: {model.bias:.2f}")
    print(f"Multi-feature Gradient Descent - Weights: {model.weights.flatten()}")
    print(f"Multi-feature Gradient Descent - Mean Squared Error: {mse_multi_gd:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(model.cost_history)), model.cost_history, color='blue')
    plt.title('Cost History during Gradient Descent (User Data)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost (MSE)')
    plt.grid(True)
    plt.show()

    model2 = LinearRegression(method="normal_equation")
    model2.fit(X, y)
    preds2 = model2.predict(X)
    mse = model2.calculate_mse(y, preds2)
    print(f"Multi-feature Normal Equation - Bias: {model2.bias:.2f}")
    print(f"Multi-feature Normal Equation - Weights: {model2.weights.flatten()}")
    print(f"Multi-feature Normal Equation - Mean Squared Error: {mse:.2f}")