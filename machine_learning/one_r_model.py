import pandas as pd


class OneR:

    def __init__(self):
        self.rule = None

    def fit(self, X, y):

        best_error = float('inf')
        best_feature = None
        best_rule = None

        for feature in X.columns:

            rule = X[feature].value_counts().idxmin()

            predictions = X[feature].map(
                lambda x: rule if x == rule else None
            )

            error = (predictions != y).sum() / len(y)

            if error < best_error:

                best_error = error
                best_feature = feature
                best_rule = rule

        self.rule = (
            best_feature,
            best_rule
        )

    def predict(self, X):

        feature, rule = self.rule

        predictions = X[feature].map(
            lambda x: rule if x == rule else None
        )

        return predictions


def predict_one_r(color):

    data = {

        'Color': ['Red', 'Green', 'Blue', 'Green', 'Red', 'Blue', 'Green'],

        'Size': ['Small', 'Large', 'Small', 'Large', 'Large', 'Small', 'Large'],

        'Shape': ['Round', 'Square', 'Round', 'Square', 'Square', 'Round', 'Round'],

        'Label': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']

    }

    df = pd.DataFrame(data)

    X = df[['Color', 'Size', 'Shape']]

    y = df['Label']

    model = OneR()

    model.fit(X, y)

    feature, rule = model.rule

    if color == rule:
        return "Yes"

    else:
        return "No"