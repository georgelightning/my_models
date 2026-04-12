# Machine Learning Models from Scratch  

This repository showcases my from-scratch implementations of two fundamental machine learning algorithms: **Linear Regression** and **Logistic Regression**. These projects are designed to demonstrate a deep understanding of how these models work under the hood, focusing on core mathematical principles and optimization techniques rather than relying on high-level libraries.

---

## Linear Regression from Scratch  
This project contains a Python implementation of a Linear Regression model. It's built to predict continuous values based on input features.

### Key Features (Linear Regression)  
- **Dual Optimization Methods**: The model can be trained using either the iterative Gradient Descent algorithm or the analytical Normal Equation, offering flexibility and a clear comparison between the two approaches.  
- **Modular Design**: Implemented as a Python class (`LinearRegression`) for reusability.  
- **Performance Tracking**: The Gradient Descent method tracks and visualizes the cost history (Mean Squared Error) over iterations, providing insight into the training process.  

### Project Purpose (Linear Regression)  
The primary goal of this implementation was to solidify my understanding of linear regression's core mechanics. By building the model without external machine learning frameworks, I gained hands-on experience with:  
- **Mathematical Foundations**: The underlying linear algebra and calculus.  
- **Optimization**: How Gradient Descent and Normal Equation determine optimal model parameters.  
- **Code Structure**: Developing a robust and understandable machine learning component.  

### How it Works (Linear Regression)  
The `LinearRegression` class handles both single and multi-feature datasets:  
- **Initialization**: Sets hyperparameters like `learning_rate` and `n_of_iterations`.  
- **Fitting (`fit()` method)**: Trains the model using either Gradient Descent or the Normal Equation. It calculates the coefficients (weights) and bias that best fit the training data.  
- **Prediction (`predict()` method)**: Uses the learned weights and bias to make predictions on new data.  
- **Evaluation (`calculate_mse()` method)**: Measures model performance using Mean Squared Error (MSE).  

The `if __name__ == "__main__":` block demonstrates training and evaluation on self-generated synthetic data, allowing for controlled testing and verification of the implementation.

---

## Logistic Regression from Scratch  
This project features a Python implementation of a Logistic Regression model, designed for binary classification tasks.

### Key Features (Logistic Regression)  
- **Binary Classification**: Predicts the probability of an input belonging to one of two classes (0 or 1).  
- **Gradient Descent Optimization**: The model is trained using the iterative Gradient Descent algorithm to minimize the cost function.  
- **Sigmoid Activation**: Employs the sigmoid function to map linear combinations of inputs to probabilities between 0 and 1.  
- **Cost Function**: Utilizes Binary Cross-Entropy Loss for measuring the error during training.  
- **Performance Tracking**: Tracks and visualizes the cost history during Gradient Descent.  

### Project Purpose (Logistic Regression)  
This implementation aimed to deepen my understanding of classification algorithms and their core components. Building Logistic Regression from scratch provided valuable insights into:  
- **Probabilistic Modeling**: How models can output probabilities for classification.  
- **Loss Functions**: The role of Binary Cross-Entropy in guiding the learning process for classification.  
- **Gradient Descent for Classification**: Applying iterative optimization to a non-linear model.  

### How it Works (Logistic Regression)  
The `LogisticRegression` class is built for binary classification:  
- **Initialization**: Sets hyperparameters for Gradient Descent.  
- **Sigmoid Function (`_sigmoid()` method)**: Transforms the linear output into a probability.  
- **Fitting (`fit()` method)**: Trains the model by iteratively updating weights and bias using Gradient Descent to minimize the Binary Cross-Entropy loss.  
- **Prediction (`predict_probabilities()` and `predict()` methods)**: Generates probabilities for the positive class and converts these into binary class labels (default threshold: 0.5).  
- **Evaluation (`calculate_accuracy()` method)**: Measures the model's performance using accuracy score.

The `if __name__ == "__main__":` block demonstrates training on synthetic data generated with `sklearn.datasets.make_classification`.

## Two-Layer Neural Network from Scratch (Python)
This project features a Python implementation of a shallow Neural Network (Multi-Layer Perceptron) using NumPy. It is built to handle multi-class classification tasks.

### Key Features (Neural Network)
- **Pure Vectorization**: The model uses matrix operations for all layers, ensuring efficient processing of large batches without the need for loops over individual samples.
- **Manual Backpropagation**: Every gradient is derived and implemented through the chain rule, providing a transparent view of how the model learns.
- **Numerical Stability**: Includes safety mechanisms like epsilon-buffering in log calculations to prevent "NaN" errors during training.

### Project Purpose (Neural Network)
This implementation was designed to bridge the gap between abstract calculus and functional code. Through this project, I deepened my understanding of:
- **Multivariable Calculus**: Implementing the partial derivatives of ReLU and Softmax functions.
- **Linear Algebra**: Handling matrix dimensions and weight initialization strategies.
- **Optimization**: Understanding how weights and biases shift during Gradient Descent to minimize Categorical Cross-Entropy.

### How it Works (Neural Network)
The `TwoLayerNN` class manages a standard Feedforward architecture:
- **Initialization**: Uses randomized weight matrices and zero-initialized biases for a 2-layer structure.
- **Forward Pass (`forward_pass()` method)**: Computes the linear combinations and activations (ReLU for hidden, Softmax for output).
- **Backpropagation (`backpropagation()` method)**: Calculates gradients for every parameter using the stored cache from the forward pass.
- **Optimization (`update_parameters()` method)**: Adjusts weights and biases using a specified learning rate.
---

## Technologies Used  
- **Python**: Core programming language.  
- **NumPy**: Efficient numerical computation and array/matrix operations.  
- **Matplotlib**: Plotting cost histories for training visualization.  
- **scikit-learn** (`make_classification`): *Only for generating synthetic data* (not for model implementation).  

---

## Getting Started  
To run these projects, follow the steps below.  

### Prerequisites  
Install required libraries:  
```bash
pip install numpy matplotlib scikit-learn
```
## Running the Code

1. **Clone this repository:**
```bash
git clone https://github.com/georgelightning/my_models.git
```
2.**Navigate to the project directory:**
```bash
cd my_models
```
3. **Run the Linear Regression model:**
```bash
python linear_regression.py
```
4. **Run the Logistic Regression model:**
```bash
python logistic_regression.py
```
5. **Run the Neural Network:**
   ```bash
python model_nn.py
```
