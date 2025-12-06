import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model on test data.

    Works with both NumPy-based and scikit-learn models.

    Args:
        model: trained classifier (either custom GaussianNaiveBayes or sklearn model)
        X_test: numpy array or pandas DataFrame of shape (n_samples, n_features) - test features
        y_test: numpy array or pandas Series of shape (n_samples,) - true labels

    Returns:
        dict: evaluation metrics including accuracy, predictions, and confusion matrix
    """
    # Convert to numpy arrays if needed (for pandas DataFrames/Series)
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = np.mean(y_pred == y_test)

    # Build confusion matrix
    classes = np.unique(y_test)
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=int)

    for true_label, pred_label in zip(y_test, y_pred):
        true_idx = np.where(classes == true_label)[0][0]
        pred_idx = np.where(classes == pred_label)[0][0]
        confusion_matrix[true_idx, pred_idx] += 1

    # Calculate per-class metrics
    class_metrics = {}
    for i, cls in enumerate(classes):
        true_positives = confusion_matrix[i, i]
        false_positives = np.sum(confusion_matrix[:, i]) - true_positives
        false_negatives = np.sum(confusion_matrix[i, :]) - true_positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

    results = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'true_labels': y_test,
        'confusion_matrix': confusion_matrix,
        'classes': classes,
        'class_metrics': class_metrics
    }

    return results


def print_evaluation_results(results):
    """
    Print evaluation results in a readable format.

    Args:
        results: dict returned by evaluate_model()
    """
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"Classes: {results['classes']}")
    print(results['confusion_matrix'])

    print(f"\nPer-Class Metrics:")
    for cls, metrics in results['class_metrics'].items():
        print(f"\nClass {cls}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

    print(f"\nCorrect predictions: {np.sum(results['predictions'] == results['true_labels'])}/{len(results['true_labels'])}")