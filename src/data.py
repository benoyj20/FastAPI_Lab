"""Data loading and splitting for the Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Column order the model is trained on. The API must send features in this order.
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
# Index in this list == class id predicted by the model.
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_data():
    """Load the Iris dataset.

    Returns:
        X (numpy.ndarray): Features in FEATURE_NAMES order.
        y (numpy.ndarray): Target labels (0=setosa, 1=versicolor, 2=virginica).
    """
    iris = load_iris()
    return iris.data, iris.target


def split_data(X, y, test_size=0.3, random_state=12):
    """Split the data into stratified training and test sets.

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
