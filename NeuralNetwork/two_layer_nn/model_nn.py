import torch


class TwoLayerNN:

    def __init__(self, n_x, n_h, n_y, learning_rate: float = 0.01, device: str = None):
        self.n_x = n_x
        self.n_y = n_y
        self.n_h = n_h
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = self._initialize_parameters()
        self.cache = {}

    def _initialize_parameters(self):
        w1 = torch.randn(self.n_h, self.n_x, device=self.device) * (2 / self.n_x) ** 0.5
        w2 = torch.randn(self.n_y, self.n_h, device=self.device) * (2 / self.n_h) ** 0.5
        b1 = torch.zeros(self.n_h, 1, device=self.device)
        b2 = torch.zeros(self.n_y, 1, device=self.device)
        return {"W1": w1, "W2": w2, "b1": b1, "b2": b2}

    @staticmethod
    def softmax(Z):
        exp_Z = torch.exp(Z - torch.max(Z, dim=0).values)
        return exp_Z / torch.sum(exp_Z, dim=0)

    @staticmethod
    def relu(Z):
        return torch.maximum(Z, torch.zeros_like(Z))

    def forward_pass(self, X):
        self.cache = {}

        Z1 = self.weights["W1"] @ X + self.weights["b1"]
        A1 = self.relu(Z1)

        Z2 = self.weights["W2"] @ A1 + self.weights["b2"]
        A2 = self.softmax(Z2)

        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2

    @staticmethod
    def cost(A2, Y):
        m = Y.shape[1]
        log_A2 = torch.log(A2 + 1e-10)
        cost = -(1 / m) * torch.sum(Y * log_A2)
        return cost.item()

    def backpropagation(self, X, Y):
        m = X.shape[1]
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]
        Z1 = self.cache["Z1"]
        W2 = self.weights["W2"]

        dZ2 = A2 - Y
        dW2 = (1 / m) * (dZ2 @ A1.T)
        db2 = (1 / m) * torch.sum(dZ2, dim=1, keepdim=True)

        dA1 = W2.T @ dZ2
        dZ1 = dA1 * (Z1 > 0).long()
        dW1 = (1 / m) * (dZ1 @ X.T)
        db1 = (1 / m) * torch.sum(dZ1, dim=1, keepdim=True)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def fit(self, X, y, epochs=10000, batch_size=None):
        X = X.to(self.device)
        y = y.to(self.device)
        costs = []

        for i in range(epochs):
            if batch_size:
                permutation = torch.randperm(X.shape[1], device=self.device)
                X = X[:, permutation]
                y = y[:, permutation]
                for j in range(0, X.shape[1], batch_size):
                    X_batch = X[:, j:j + batch_size]
                    y_batch = y[:, j:j + batch_size]
                    A2 = self.forward_pass(X_batch)
                    grads = self.backpropagation(X_batch, y_batch)
                    self.update_parameters(grads)
                cost = self.cost(self.forward_pass(X), y)
            else:
                A2 = self.forward_pass(X)
                cost = self.cost(A2, y)
                grads = self.backpropagation(X, y)
                self.update_parameters(grads)

            if i % 1000 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost:.4f}")

        return self.weights, costs

    def update_parameters(self, grads):
        alpha = self.learning_rate
        self.weights["W1"] -= alpha * grads["dW1"]
        self.weights["b1"] -= alpha * grads["db1"]
        self.weights["W2"] -= alpha * grads["dW2"]
        self.weights["b2"] -= alpha * grads["db2"]

    def predict_accuracy(self, X, y):
        X = X.to(self.device)
        y = y.to(self.device)
        A2 = self.forward_pass(X)
        predictions = torch.argmax(A2, dim=0)
        true_labels = torch.argmax(y, dim=0)
        return (predictions == true_labels).float().mean().item()
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        A2 = self.forward_pass(X)
        return torch.argmax(A2, dim=0).cpu().numpy()