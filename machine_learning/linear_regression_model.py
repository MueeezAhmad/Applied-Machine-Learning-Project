import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def run_linear_regression(filepath):

    df = pd.read_csv(filepath)

    numeric_data = df.select_dtypes(include=['int64', 'float64'])

    rows = df.shape[0]
    columns = df.shape[1]

    target_column = numeric_data.columns[-1]

    X = numeric_data.drop(columns=[target_column])

    y = numeric_data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    error = mean_absolute_error(
        y_test,
        predictions
    )

    plt.figure(figsize=(6,4))

    plt.scatter(y_test, predictions)

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")

    plt.title("Linear Regression")

    plt.savefig("static/regression.png")

    plt.close()

    return {
        "rows": rows,
        "columns": columns,
        "target_column": target_column,
        "error": round(error,2)
    }