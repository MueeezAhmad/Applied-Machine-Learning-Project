import speech_recognition as sr
import pyttsx3

from transformers import pipeline


qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-cased-distilled-squad"
)

engine = pyttsx3.init()


def answer_question(audio_path, context):

    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:

        audio = recognizer.record(source)

    question = recognizer.recognize_google(audio)

    print("Question:", question)

    result = qa_pipeline(

        question=question,

        context=context

    )

    answer = result["answer"]

    engine = pyttsx3.init(driverName='nsss')

    engine.say(answer)

    engine.runAndWait()

    engine.stop()

    return {

        "question": question,

        "answer": answer

    }