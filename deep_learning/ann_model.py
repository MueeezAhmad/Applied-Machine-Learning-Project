import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = Sequential([

    Dense(
        8,
        activation='relu',
        input_shape=(4,)
    ),

    Dense(
        3,
        activation='softmax'
    )

])


model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)


model.fit(

    X_train,
    y_train,

    epochs=50,

    verbose=0

)


def predict_flower_ann(

        sepal_length,
        sepal_width,
        petal_length,
        petal_width
):

    sample = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(
        sample,
        verbose=0
    )

    class_index = np.argmax(
        prediction
    )

    return iris.target_names[
        class_index
    ]