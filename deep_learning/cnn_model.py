from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing import image
import numpy as np


model = models.Sequential([

    layers.Input(shape=(128,128,3)),

    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dense(1, activation='sigmoid')

])

model.load_weights("gender_weights.weights.h5")


def predict_gender(img_path):

    img = image.load_img(
        img_path,
        target_size=(128,128)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    score = prediction[0][0]

    if score > 0.5:

        return f"Male"

    else:

        return f"Female"