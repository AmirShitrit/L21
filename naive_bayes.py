import numpy as np


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier implemented with only NumPy.

    Assumes features follow a Gaussian (normal) distribution.
    """

    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_vars = {}

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.

        Args:
            X: numpy array of shape (n_samples, n_features) - training features
            y: numpy array of shape (n_samples,) - training labels
        """
        self.classes = np.unique(y)
        n_samples = X.shape[0]

        # Calculate prior probabilities, means, and variances for each class
        for cls in self.classes:
            # Get all samples belonging to this class
            X_cls = X[y == cls]

            # Prior probability: P(class)
            self.class_priors[cls] = X_cls.shape[0] / n_samples

            # Mean of each feature for this class
            self.class_means[cls] = np.mean(X_cls, axis=0)

            # Variance of each feature for this class
            self.class_vars[cls] = np.var(X_cls, axis=0)

    def _calculate_likelihood(self, x, mean, var):
        """
        Calculate Gaussian probability density function.

        P(x|class) = (1 / sqrt(2 * pi * var)) * exp(-(x - mean)^2 / (2 * var))
        """
        eps = 1e-6  # Small constant to avoid division by zero
        var = var + eps

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def _calculate_posterior(self, x, cls):
        """
        Calculate posterior probability for a given class.

        Using log probabilities to avoid numerical underflow:
        log P(class|x) = log P(class) + sum(log P(x_i|class))
        """
        # Start with prior probability (in log space)
        log_prior = np.log(self.class_priors[cls])

        # Calculate likelihood for each feature (in log space)
        likelihood = self._calculate_likelihood(x, self.class_means[cls], self.class_vars[cls])
        log_likelihood = np.sum(np.log(likelihood))

        # Posterior = Prior * Likelihood (in log space: addition)
        return log_prior + log_likelihood

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of predicted class labels
        """
        predictions = []

        for x in X:
            # Calculate posterior for each class
            posteriors = {cls: self._calculate_posterior(x, cls) for cls in self.classes}

            # Predict the class with highest posterior probability
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.

        Args:
            X: numpy array of shape (n_samples, n_features)

        Returns:
            numpy array of shape (n_samples, n_classes) with probabilities
        """
        probabilities = []

        for x in X:
            # Calculate posterior for each class (log space)
            log_posteriors = {cls: self._calculate_posterior(x, cls) for cls in self.classes}

            # Convert back from log space and normalize
            posteriors = {cls: np.exp(log_post) for cls, log_post in log_posteriors.items()}
            total = sum(posteriors.values())

            # Normalize to get probabilities
            probs = [posteriors[cls] / total for cls in self.classes]
            probabilities.append(probs)

        return np.array(probabilities)


def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier.

    Args:
        X_train: numpy array of shape (n_samples, n_features) - training features
        y_train: numpy array of shape (n_samples,) - training labels

    Returns:
        GaussianNaiveBayes: trained classifier
    """
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)
    return model