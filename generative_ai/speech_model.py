import speech_recognition as sr


def analyze_voice(audio_path):

    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:

        audio_data = recognizer.record(source)

    text = recognizer.recognize_google(
        audio_data,
        language="en-US"
    )

    return text