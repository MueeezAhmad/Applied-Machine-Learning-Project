from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
knn = KNeighborsClassifier(n_neighbors=3)

# Train model
knn.fit(X_train, y_train)


def predict_flower(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width):

    prediction = knn.predict([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])[0]

    classes = iris.target_names

    return classes[prediction]