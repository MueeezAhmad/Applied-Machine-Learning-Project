import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


def run_dbscan(filepath):

    df = pd.read_csv(filepath)

    numeric_data = df.select_dtypes(
        include=['int64', 'float64']
    )

    data = numeric_data.iloc[:, :2]

    model = DBSCAN(
        eps=5,
        min_samples=5
    )

    labels = model.fit_predict(data)

    plt.figure(figsize=(6,4))

    plt.scatter(
        data.iloc[:,0],
        data.iloc[:,1],
        c=labels
    )

    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])

    plt.title("DBSCAN")

    plt.savefig("static/dbscan.png")

    plt.close()