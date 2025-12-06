import numpy as np
import matplotlib.pyplot as plt


def visualize_model_comparison(numpy_results, sklearn_results, output_file='model_comparison.png'):
    """
    Visualize comparison between NumPy and scikit-learn model results.

    Args:
        numpy_results: dict returned by evaluate_model() for NumPy model
        sklearn_results: dict returned by evaluate_model() for sklearn model
        output_file: path to save the visualization (default: 'model_comparison.png')
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Naive Bayes Comparison: NumPy vs Scikit-Learn', fontsize=16, fontweight='bold')

    # Plot confusion matrices
    im1 = axes[0, 0].imshow(numpy_results['confusion_matrix'], cmap='Blues', aspect='auto')
    axes[0, 0].set_title('NumPy - Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('True')
    axes[0, 0].set_xticks(range(len(numpy_results['classes'])))
    axes[0, 0].set_yticks(range(len(numpy_results['classes'])))
    axes[0, 0].set_xticklabels(numpy_results['classes'], rotation=45)
    axes[0, 0].set_yticklabels(numpy_results['classes'])
    for i in range(len(numpy_results['classes'])):
        for j in range(len(numpy_results['classes'])):
            axes[0, 0].text(j, i, numpy_results['confusion_matrix'][i, j],
                           ha='center', va='center', color='red', fontweight='bold')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[1, 0].imshow(sklearn_results['confusion_matrix'], cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Scikit-learn - Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    axes[1, 0].set_xticks(range(len(sklearn_results['classes'])))
    axes[1, 0].set_yticks(range(len(sklearn_results['classes'])))
    axes[1, 0].set_xticklabels(sklearn_results['classes'], rotation=45)
    axes[1, 0].set_yticklabels(sklearn_results['classes'])
    for i in range(len(sklearn_results['classes'])):
        for j in range(len(sklearn_results['classes'])):
            axes[1, 0].text(j, i, sklearn_results['confusion_matrix'][i, j],
                           ha='center', va='center', color='red', fontweight='bold')
    plt.colorbar(im2, ax=axes[1, 0])

    # Plot per-class metrics for NumPy
    classes = list(numpy_results['class_metrics'].keys())
    metrics = ['precision', 'recall', 'f1_score']
    x = np.arange(len(classes))
    width = 0.25

    for idx, metric in enumerate(metrics):
        values = [numpy_results['class_metrics'][cls][metric] for cls in classes]
        axes[0, 1].bar(x + idx * width, values, width, label=metric.capitalize())

    axes[0, 1].set_title('NumPy - Per-Class Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_xticks(x + width)
    axes[0, 1].set_xticklabels(classes, rotation=45)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1.1])
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Plot per-class metrics for sklearn
    for idx, metric in enumerate(metrics):
        values = [sklearn_results['class_metrics'][cls][metric] for cls in classes]
        axes[1, 1].bar(x + idx * width, values, width, label=metric.capitalize())

    axes[1, 1].set_title('Scikit-learn - Per-Class Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(classes, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Plot accuracy comparison
    models = ['NumPy', 'Scikit-learn']
    accuracies = [numpy_results['accuracy'], sklearn_results['accuracy']]
    colors = ['steelblue', 'coral']

    axes[0, 2].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Accuracy Comparison')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].set_ylim([0, 1.1])
    axes[0, 2].grid(axis='y', alpha=0.3)
    for i, (model, acc) in enumerate(zip(models, accuracies)):
        axes[0, 2].text(i, acc + 0.02, f'{acc:.4f}', ha='center', fontweight='bold')

    # Plot prediction agreement
    agreement = np.sum(numpy_results['predictions'] == sklearn_results['predictions'])
    disagreement = len(numpy_results['predictions']) - agreement

    axes[1, 2].pie([agreement, disagreement], labels=['Agree', 'Disagree'],
                   autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    axes[1, 2].set_title(f'Prediction Agreement\n({agreement}/{len(numpy_results["predictions"])} samples)')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")

    return output_file