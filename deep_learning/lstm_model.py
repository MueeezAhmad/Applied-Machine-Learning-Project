import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense


# Training data

sentences = [

    # Positive
    "I love this movie",
    "This movie was amazing",
    "The film was fantastic",
    "Great acting and storyline",
    "A wonderful experience",
    "The best movie ever",
    "I enjoyed every moment",
    "Absolutely brilliant film",
    "A masterpiece of cinema",
    "Excellent direction and acting",
    "This movie was inspiring",
    "Highly recommended movie",
    "Superb performance by actors",
    "The story was beautiful",
    "Fantastic visuals and music",
    "I liked this film very much",
    "It was entertaining and fun",
    "An unforgettable experience",
    "One of my favorite movies",
    "Amazing and emotional film",

    # Negative
    "I hate this movie",
    "The movie was terrible",
    "This film was boring",
    "I did not enjoy the movie",
    "Absolutely awful film",
    "Bad acting and weak story",
    "The plot was dull",
    "I regret watching this movie",
    "Very disappointing movie",
    "Worst movie ever",
    "I would not recommend this film",
    "Poor direction and acting",
    "This movie was a waste of time",
    "I disliked the entire film",
    "Terrible experience",
    "The story was confusing",
    "Not worth watching",
    "It was very boring",
    "The movie was horrible",
    "Completely disappointing"

]


# Labels
# Positive = 1
# Negative = 0

labels = np.array([

    1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,

    0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0

])


# Tokenizer

tokenizer = Tokenizer(num_words=10000)

tokenizer.fit_on_texts(
    sentences
)

sequences = tokenizer.texts_to_sequences(
    sentences
)


# Padding

maxlen = 10

data = pad_sequences(
    sequences,
    maxlen=maxlen
)


# Build GRU Model

model = Sequential()

model.add(

    Embedding(
        input_dim=10000,
        output_dim=128
    )

)

model.add(

    GRU(
        10
    )

)

model.add(

    Dense(
        1,
        activation='sigmoid'
    )

)


# Compile

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)


# Train

model.fit(

    data,

    labels,

    epochs=50,

    batch_size=2,

    verbose=0

)


# Check Accuracy

loss, accuracy = model.evaluate(

    data,

    labels,

    verbose=0

)

print("Training Accuracy:", round(accuracy, 2))


# Prediction Function

def predict_sentiment(text):

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(

        sequence,

        maxlen=maxlen

    )

    prediction = model.predict(

        padded,

        verbose=0

    )[0][0]


    if prediction > 0.5:

        return f"Positive"

    else:

        return f"Negative"