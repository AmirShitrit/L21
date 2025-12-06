# Product Requirements Document (PRD)
## Iris Classification: NumPy vs Scikit-Learn Naive Bayes Comparison

---

## 1. Overview

### 1.1 Purpose
Develop a machine learning comparison tool that implements Gaussian Naive Bayes classification from scratch using NumPy and compares it against scikit-learn's implementation on the Iris dataset.

### 1.2 Objectives
- Implement a complete machine learning pipeline for iris species classification
- Compare custom NumPy-based implementation with industry-standard scikit-learn
- Provide comprehensive evaluation metrics and visual comparisons
- Demonstrate understanding of Naive Bayes algorithm fundamentals

---

## 2. Functional Requirements

### 2.1 Data Management

#### FR-1: Dataset Acquisition
- **Requirement**: Download and store the Iris dataset locally
- **Format**: CSV file with 150 samples, 4 features (sepal length, sepal width, petal length, petal width), and species labels
- **Source**: UCI Machine Learning Repository
- **Output**: `iris.csv`

#### FR-2: Data Splitting
- **Requirement**: Split dataset into training (75%) and testing (25%) sets
- **Method**: Random sampling with seed for reproducibility
- **Implementation**: Custom function without external ML libraries
- **Output**: Separate training and testing dataframes

### 2.2 Model Training

#### FR-3: NumPy-Based Naive Bayes Implementation
- **Requirement**: Implement Gaussian Naive Bayes classifier using only NumPy
- **Features**:
  - Calculate prior probabilities for each class
  - Compute mean and variance for each feature per class
  - Implement Gaussian probability density function
  - Use log probabilities to prevent numerical underflow
  - Support both prediction and probability estimation
- **Output**: Trained `GaussianNaiveBayes` model

#### FR-4: Scikit-Learn Naive Bayes Implementation
- **Requirement**: Train Gaussian Naive Bayes using scikit-learn's `GaussianNB`
- **Purpose**: Benchmark comparison against industry standard
- **Output**: Trained sklearn `GaussianNB` model

### 2.3 Model Evaluation

#### FR-5: Comprehensive Evaluation Metrics
- **Requirement**: Evaluate both models with identical metrics
- **Metrics**:
  - Overall accuracy
  - Confusion matrix
  - Per-class precision, recall, and F1-score
  - Prediction vs true label comparison
- **Implementation**: Model-agnostic evaluation function supporting both NumPy and sklearn models

#### FR-6: Model Comparison
- **Requirement**: Direct comparison between NumPy and sklearn implementations
- **Comparisons**:
  - Accuracy difference
  - Prediction agreement percentage
  - Side-by-side metric display

### 2.4 Visualization

#### FR-7: Visual Comparison Dashboard
- **Requirement**: Generate comprehensive visualization comparing both models
- **Components**:
  1. **Confusion Matrices** (2 plots): One for each implementation with annotated values
  2. **Per-Class Metrics** (2 plots): Bar charts showing precision, recall, F1-score for each iris species
  3. **Accuracy Comparison** (1 plot): Bar chart comparing overall model accuracy
  4. **Prediction Agreement** (1 plot): Pie chart showing agreement/disagreement between models
- **Format**: High-resolution PNG (300 DPI)
- **Layout**: 2x3 grid layout

---

## 3. Non-Functional Requirements

### 3.1 Code Organization
- **Modularity**: Separate files for distinct functionality
  - `data_utils.py`: Data loading and splitting
  - `naive_bayes.py`: NumPy-based implementation
  - `naive_bayes_sklearn.py`: Scikit-learn wrapper
  - `evaluate.py`: Model evaluation functions
  - `visualize.py`: Visualization functions
  - `main.py`: Orchestration and execution
- **Reusability**: Functions should be reusable and well-documented
- **Separation of Concerns**: Clear boundaries between data processing, training, evaluation, and visualization

### 3.2 Performance
- **Training Time**: Near-instantaneous for both implementations on Iris dataset
- **Memory Efficiency**: Minimal memory footprint suitable for educational purposes

### 3.3 Maintainability
- **Documentation**: Docstrings for all functions with parameter and return descriptions
- **Code Style**: Clean, readable Python code
- **Type Hints**: Clear parameter types in function signatures

### 3.4 Dependencies
- **Required Libraries**:
  - `numpy`: Numerical computations and custom implementation
  - `pandas`: Data loading and manipulation
  - `matplotlib`: Visualization
  - `scikit-learn`: Benchmark implementation
- **Python Version**: 3.10+

---

## 4. Technical Architecture

### 4.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                   (Orchestration Layer)                      │
└─────────────────────────────────────────────────────────────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
    ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐
    │data_utils │  │  naive_   │  │ evaluate  │  │visualize │
    │    .py    │  │  bayes    │  │   .py     │  │   .py    │
    │           │  │  .py/.py  │  │           │  │          │
    └───────────┘  └───────────┘  └───────────┘  └──────────┘
```

### 4.2 Data Flow

1. **Data Acquisition**: Download iris.csv → Store locally
2. **Data Preparation**: Load CSV → Split 75/25 → Create train/test sets
3. **Training**: Train NumPy model → Train sklearn model
4. **Evaluation**: Evaluate NumPy model → Evaluate sklearn model
5. **Comparison**: Calculate differences → Generate metrics
6. **Visualization**: Create plots → Save to file

### 4.3 Module Specifications

#### data_utils.py
- `split_data(df, train_ratio, random_state)`: Random train/test split

#### naive_bayes.py
- `GaussianNaiveBayes` class:
  - `fit(X, y)`: Train the model
  - `predict(X)`: Predict class labels
  - `predict_proba(X)`: Predict class probabilities
- `train_naive_bayes(X_train, y_train)`: Convenience wrapper

#### naive_bayes_sklearn.py
- `train_naive_bayes_sklearn(X_train, y_train)`: Sklearn wrapper

#### evaluate.py
- `evaluate_model(model, X_test, y_test)`: Calculate all metrics
- `print_evaluation_results(results)`: Format and display results

#### visualize.py
- `visualize_model_comparison(numpy_results, sklearn_results, output_file)`: Generate comparison plots

---

## 5. Deliverables

### 5.1 Code Files
- ✅ `iris.csv` - Dataset
- ✅ `data_utils.py` - Data splitting utilities
- ✅ `naive_bayes.py` - Custom NumPy implementation
- ✅ `naive_bayes_sklearn.py` - Scikit-learn wrapper
- ✅ `evaluate.py` - Evaluation functions
- ✅ `visualize.py` - Visualization functions
- ✅ `main.py` - Main execution script

### 5.2 Output Files
- ✅ `model_comparison.png` - Visual comparison dashboard
- ✅ `PROMPTS.md` - Development history log
- ✅ `PRD.md` - This document

### 5.3 Console Output
- Dataset loading confirmation
- Training progress indicators
- Detailed evaluation metrics for both models
- Accuracy comparison summary
- File save confirmations

---

## 6. Success Criteria

### 6.1 Functional Success
- ✅ Both models successfully train on Iris dataset
- ✅ Both models achieve >90% accuracy (expected for Iris)
- ✅ Evaluation metrics are calculated correctly
- ✅ Visualization is generated successfully

### 6.2 Technical Success
- ✅ NumPy implementation produces similar results to scikit-learn
- ✅ Code is modular and well-organized
- ✅ All functions have proper documentation
- ✅ No external ML libraries used in custom implementation (except NumPy)

### 6.3 Educational Success
- ✅ Demonstrates understanding of Naive Bayes algorithm
- ✅ Shows ability to implement ML algorithms from scratch
- ✅ Provides clear comparison between implementations
- ✅ Produces publication-quality visualizations

---

## 7. Future Enhancements

### 7.1 Potential Improvements
1. **Additional Algorithms**: Implement other classifiers (KNN, Decision Trees, SVM)
2. **Cross-Validation**: Add k-fold cross-validation for more robust evaluation
3. **Hyperparameter Tuning**: Explore different smoothing parameters
4. **Feature Engineering**: Add feature scaling and selection
5. **Interactive Visualization**: Web-based dashboard with plotly/dash
6. **Model Persistence**: Save/load trained models with pickle
7. **CLI Interface**: Command-line arguments for different configurations
8. **Unit Tests**: Comprehensive test suite for all functions

### 7.2 Scalability
- Support for larger datasets
- Parallel training for multiple models
- Batch prediction capabilities
- Integration with MLOps tools

---

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-06 | Generated from prompts | Initial PRD based on development session |

---

## 9. Appendix

### 9.1 Algorithm Description

**Gaussian Naive Bayes** assumes:
1. Features are conditionally independent given the class
2. Features follow a Gaussian (normal) distribution
3. Prior probabilities are calculated from training data frequency

**Formula**:
```
P(class|X) ∝ P(class) × ∏ P(x_i|class)

where:
P(x_i|class) = (1/√(2πσ²)) × exp(-(x_i - μ)² / (2σ²))
```

### 9.2 Dataset Information

**Iris Dataset**:
- **Samples**: 150 (50 per class)
- **Features**: 4 continuous measurements
- **Classes**: 3 iris species (setosa, versicolor, virginica)
- **Source**: Fisher's 1936 paper
- **Use Case**: Multi-class classification benchmark

### 9.3 Key Design Decisions

1. **Random Splitting**: Used pandas `.sample()` for simplicity and reproducibility
2. **Log Probabilities**: Prevents numerical underflow in NumPy implementation
3. **Model-Agnostic Evaluation**: Single evaluation function works with both implementations
4. **Modular Visualization**: Separate file allows easy reuse for other projects
5. **No Data Preprocessing**: Iris dataset is clean; no normalization needed for Naive Bayes
