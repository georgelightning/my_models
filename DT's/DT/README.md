# C++ Decision Tree Classifier (from scratch)

A lightweight, high-performance Decision Tree implementation written in C++17. This project implements a classification engine from the ground up, utilizing **Shannon Entropy** and **Information Gain** to perform recursive binary splitting on numerical datasets.

## Key Features
* **Custom Dataset Loader:** Handles CSV parsing, string-to-numeric conversion, and automatic label encoding for categorical targets.
* **Recursive ID3/C4.5 Logic:** Implements the classic top-down induction of decision trees.
* **Efficient Splitting:** Uses a "Sort & Slide" mechanism. By sorting features at each node, the engine finds optimal thresholds in $O(numFeatures \cdot n \log n)$ time.
* **Memory Management:** Features a recursive destructor to ensure all dynamically allocated `Node` pointers are cleaned up, preventing memory leaks.
* **Math-First Approach:** Implemented with a focus on matrix-like data handling and logarithmic entropy calculations.

## Data Requirements

To ensure the engine runs successfully, your input data should follow these rules:

1. **Numerical Features:** All independent variables (the columns before the last one) must be numeric (integers or doubles).
2. **Categorical Target:** The final column (the label) can be categorical (e.g., "Iris-setosa", "Win", "Loss"). The `DatasetLoader` will automatically map these to unique integer IDs.
3. **Clean Data:** The CSV should not contain missing values or non-numeric characters within the feature columns, as the parser uses `std::stod`.

## Performance Example: Iris Dataset

The engine was tested using the standard Iris Flower dataset (150 samples, 4 features, 3 classes).

* **Training/Test Split:** 80% / 20%
* **Max Depth:** 10 (Adjustable in `main.cpp`)
* **Results:** Consistently achieves ~90-95% Accuracy.

## How to Use

### 1. Prepare your CSV
Ensure your data is in a standard CSV format without trailing commas:
```csv
5.1,3.5,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor