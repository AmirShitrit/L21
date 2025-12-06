from sklearn.naive_bayes import GaussianNB


def train_naive_bayes_sklearn(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier using scikit-learn.

    Args:
        X_train: numpy array or pandas DataFrame of shape (n_samples, n_features) - training features
        y_train: numpy array or pandas Series of shape (n_samples,) - training labels

    Returns:
        GaussianNB: trained scikit-learn Gaussian Naive Bayes classifier
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model