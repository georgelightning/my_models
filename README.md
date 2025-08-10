Machine Learning Models from Scratch
This repository showcases my from-scratch implementations of two fundamental machine learning algorithms: Linear Regression and Logistic Regression. These projects are designed to demonstrate a deep understanding of how these models work under the hood, focusing on core mathematical principles and optimization techniques rather than relying on high-level libraries.

Linear Regression from Scratch
This project contains a Python implementation of a Linear Regression model. It's built to predict continuous values based on input features.

Key Features (Linear Regression)
Dual Optimization Methods: The model can be trained using either the iterative Gradient Descent algorithm or the analytical Normal Equation, offering flexibility and a clear comparison between the two approaches.

Modular Design: Implemented as a Python class (LinearRegression) for reusability.

Performance Tracking: The Gradient Descent method tracks and visualizes the cost history (Mean Squared Error) over iterations, providing insight into the training process.

Project Purpose (Linear Regression)
The primary goal of this implementation was to solidify my understanding of linear regression's core mechanics. By building the model without external machine learning frameworks, I gained hands-on experience with:

Mathematical Foundations: The underlying linear algebra and calculus.

Optimization: How Gradient Descent and Normal Equation determine optimal model parameters.

Code Structure: Developing a robust and understandable machine learning component.

How it Works (Linear Regression)
The LinearRegression class handles both single and multi-feature datasets.

Initialization: Sets hyperparameters like learning_rate and n_of_iterations.

Fitting (fit() method): Trains the model using either Gradient Descent or the Normal Equation. It calculates the coefficients (weights) and bias that best fit the training data.

Prediction (predict() method): Uses the learned weights and bias to make predictions on new data.

Evaluation (calculate_mse() method): Measures model performance using Mean Squared Error (MSE).

The if __name__ == "__main__": block demonstrates training and evaluation on self-generated synthetic data, allowing for controlled testing and verification of the implementation.

Logistic Regression from Scratch
This project features a Python implementation of a Logistic Regression model, designed for binary classification tasks.

Key Features (Logistic Regression)
Binary Classification: Predicts the probability of an input belonging to one of two classes (0 or 1).

Gradient Descent Optimization: The model is trained using the iterative Gradient Descent algorithm to minimize the cost function.

Sigmoid Activation: Employs the sigmoid function to map linear combinations of inputs to probabilities between 0 and 1.

Cost Function: Utilizes Binary Cross-Entropy Loss for measuring the error during training.

Performance Tracking: Tracks and visualizes the cost history during Gradient Descent.

Project Purpose (Logistic Regression)
This implementation aimed to deepen my understanding of classification algorithms and their core components. Building Logistic Regression from scratch provided valuable insights into:

Probabilistic Modeling: How models can output probabilities for classification.

Loss Functions: The role of Binary Cross-Entropy in guiding the learning process for classification.

Gradient Descent for Classification: Applying iterative optimization to a non-linear model.

How it Works (Logistic Regression)
The LogisticRegression class is built for binary classification:

Initialization: Sets hyperparameters for Gradient Descent.

Sigmoid Function (_sigmoid() method): Transforms the linear output into a probability.

Fitting (fit() method): Trains the model by iteratively updating weights and bias using Gradient Descent to minimize the Binary Cross-Entropy loss.

Prediction (predict_probabilities() and predict() methods): Generates probabilities for the positive class and then converts these probabilities into binary class labels based on a threshold (default 0.5).

Evaluation (calculate_accuracy() method): Measures the model's performance using accuracy score.

The if __name__ == "__main__": block demonstrates how to generate synthetic classification data using sklearn.datasets.make_classification and then train, predict, and evaluate the Logistic Regression model.

Technologies Used
Python: The core programming language for both models.

NumPy: Essential for efficient numerical computation and array/matrix operations crucial for both algorithms.

Matplotlib: Used for plotting cost histories, providing visual insights into the training process of both models.

scikit-learn (make_classification): Utilized solely for generating synthetic classification data for testing the Logistic Regression model, not for the model implementation itself.

Getting Started
To run these projects, follow the steps below.

Prerequisites
You will need the following Python libraries installed:

pip install numpy matplotlib scikit-learn

Running the Code
Assuming you have saved the Linear Regression code in linear_regression.py and the Logistic Regression code in logistic_regression.py:

Clone this repository:

git clone https://github.com/your-username/your-repo-name.git

Navigate to the project directory:

cd your-repo-name

Run the Linear Regression model:

python linear_regression.py

This will execute the linear regression training and display its results, including a plot of the cost history.

Run the Logistic Regression model:

python logistic_regression.py

This will execute the logistic regression training and display its results, including a plot of the cost history.
