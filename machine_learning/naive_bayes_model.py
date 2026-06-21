import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


data = {

    "Email_Text": [

        "Call now and claim your free lottery prize!",
        "Meeting schedule has been updated.",
        "Urgent: click link to reset your account.",
        "Project report is due next week.",
        "Win cash prizes and a cruise ship ticket.",
        "I hope this email finds you well.",
        "Money back guarantee no hidden fees.",
        "Review the attached document for Q3.",
        "Limited time offer: get rich quick!",
        "Can we reschedule our sync up call?"

    ],

    "Label": [1,0,1,0,1,0,1,0,1,0]

}

df = pd.DataFrame(data)

X = df["Email_Text"]

y = df["Label"]

vectorizer = CountVectorizer()

X_features = vectorizer.fit_transform(X)

model = MultinomialNB()

model.fit(X_features, y)


def classify_email(email):

    email_features = vectorizer.transform([email])

    prediction = model.predict(email_features)[0]

    if prediction == 1:
        return "Spam"

    else:
        return "Ham"