import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


def run_kmeans(filepath):

    df = pd.read_csv(filepath)

    numeric_data = df.select_dtypes(
        include=['int64', 'float64']
    )

    data = numeric_data.iloc[:, :2]

    model = KMeans(
        n_clusters=3,
        random_state=42
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

    plt.title("KMeans")

    plt.savefig("static/kmeans.png")

    plt.close()