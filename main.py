import pandas as pd
from data_utils import split_data
from naive_bayes import train_naive_bayes
from naive_bayes_sklearn import train_naive_bayes_sklearn
from evaluate import evaluate_model, print_evaluation_results
from visualize import visualize_model_comparison

# Load the iris dataset
df = pd.read_csv('iris.csv')

print("=" * 70)
print("IRIS CLASSIFICATION: NumPy vs Scikit-Learn Naive Bayes Comparison")
print("=" * 70)

print(f"\nDataset loaded successfully!")
print(f"Total samples: {len(df)}")

# Split into training and testing
train_df, test_df = split_data(df, train_ratio=0.75, random_state=42)

print(f"\nTraining set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Testing set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Prepare features and labels
X_train = train_df.drop('species', axis=1).values
y_train = train_df['species'].values
X_test = test_df.drop('species', axis=1).values
y_test = test_df['species'].values

# Train NumPy-based model
print("\n" + "=" * 70)
print("1. TRAINING NUMPY-BASED NAIVE BAYES MODEL")
print("=" * 70)
numpy_model = train_naive_bayes(X_train, y_train)
print("NumPy model trained successfully!")

# Evaluate NumPy model
numpy_results = evaluate_model(numpy_model, X_test, y_test)
print("\nNumPy Model Evaluation:")
print_evaluation_results(numpy_results)

# Train sklearn model
print("\n" + "=" * 70)
print("2. TRAINING SCIKIT-LEARN NAIVE BAYES MODEL")
print("=" * 70)
sklearn_model = train_naive_bayes_sklearn(X_train, y_train)
print("Scikit-learn model trained successfully!")

# Evaluate sklearn model
sklearn_results = evaluate_model(sklearn_model, X_test, y_test)
print("\nScikit-learn Model Evaluation:")
print_evaluation_results(sklearn_results)

# Compare models
print("\n" + "=" * 70)
print("3. MODEL COMPARISON")
print("=" * 70)
print(f"\nNumPy Model Accuracy:        {numpy_results['accuracy']:.4f} ({numpy_results['accuracy']*100:.2f}%)")
print(f"Scikit-learn Model Accuracy: {sklearn_results['accuracy']:.4f} ({sklearn_results['accuracy']*100:.2f}%)")
print(f"Accuracy Difference:         {abs(numpy_results['accuracy'] - sklearn_results['accuracy']):.4f}")

# Visualize comparison
print("\n" + "=" * 70)
visualize_model_comparison(numpy_results, sklearn_results, 'model_comparison.png')
print("=" * 70)
