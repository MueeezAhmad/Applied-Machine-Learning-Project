import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def run_apriori(filepath):

    df = pd.read_csv(filepath)

    transactions = []

    for items in df["Items"]:

        transactions.append(
            items.split(",")
        )

    te = TransactionEncoder()

    te_array = te.fit(transactions).transform(transactions)

    basket = pd.DataFrame(
        te_array,
        columns=te.columns_
    )

    frequent_itemsets = apriori(

        basket,

        min_support=0.4,

        use_colnames=True

    )

    rules = association_rules(

        frequent_itemsets,

        metric="confidence",

        min_threshold=0.6

    )

    return rules[
        ['antecedents',
         'consequents',
         'support',
         'confidence']
    ].to_string()