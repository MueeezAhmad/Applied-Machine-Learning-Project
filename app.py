from flask import Flask, render_template, request
import os

#machine learning imports
from machine_learning.naive_bayes_model import classify_email
from machine_learning.knn_model import predict_flower
from machine_learning.linear_regression_model import run_linear_regression
from machine_learning.kmeans_model import run_kmeans
from machine_learning.dbscan_model import run_dbscan
from machine_learning.one_r_model import predict_one_r

#deep learning imports
from deep_learning.ann_model import predict_flower_ann
from deep_learning.cnn_model import predict_gender
from deep_learning.lstm_model import predict_sentiment

#generative_ai imports
from generative_ai.sentiment_model import analyze_sentiment
from generative_ai.translation_model import translate_text
from generative_ai.speech_model import analyze_voice
from generative_ai.qa_model import answer_question
from generative_ai.ner_model import extract_entities
from generative_ai.text_generation_model import generate_text
from generative_ai.zero_shot_model import classify_text

#apriori
from association_rules.apriori_model import run_apriori

app = Flask(__name__)


# ==========================
# HOME
# ==========================

@app.route('/')
def home():
    return render_template('home.html')


# ==========================
# MAIN PAGES
# ==========================

@app.route('/machine_learning')
def machine_learning():
    return render_template(
        'machine_learning/machine_learning.html'
    )


@app.route('/deep_learning')
def deep_learning():
    return render_template(
        'deep_learning/deep_learning.html'
    )


@app.route('/generative_ai')
def generative_ai():
    return render_template(
        'generative_ai/generative_ai.html'
    )


@app.route('/apriori')
def apriori():
    return render_template(
        'association_rules/apriori.html'
    )


# ==========================
# MACHINE LEARNING
# ==========================

@app.route('/naive_bayes')
def naive_bayes():
    return render_template(
        'machine_learning/naive_bayes.html'
    )


@app.route('/knn')
def knn():
    return render_template(
        'machine_learning/knn.html'
    )


@app.route('/linear_regression')
def linear_regression():
    return render_template(
        'machine_learning/linear_regression.html'
    )


@app.route('/kmeans')
def kmeans():
    return render_template(
        'machine_learning/kmeans.html'
    )


@app.route('/dbscan')
def dbscan():
    return render_template(
        'machine_learning/dbscan.html'
    )


@app.route('/one_r')
def one_r():
    return render_template(
        'machine_learning/one_r.html'
    )

# ==========================
# DEEP LEARNING
# ==========================

@app.route('/ann')
def ann():
    return render_template(
        'deep_learning/ann.html'
    )


@app.route('/cnn')
def cnn():
    return render_template(
        'deep_learning/cnn.html'
    )


@app.route('/lstm')
def lstm():
    return render_template(
        'deep_learning/lstm.html'
    )


# ==========================
# GENERATIVE AI
# ==========================

@app.route('/sentiment')
def sentiment():
    return render_template(
        'generative_ai/sentiment.html'
    )

@app.route('/translation')
def translation():
    return render_template(
        'generative_ai/translation.html'
    )

@app.route('/speech')
def speech():
    return render_template(
        'generative_ai/speech.html'
    )


@app.route('/qa')
def qa():
    return render_template(
        'generative_ai/qa.html'
    )


@app.route('/ner')
def ner():
    return render_template(
        'generative_ai/ner.html'
    )


@app.route('/text_generation')
def text_generation():
    return render_template(
        'generative_ai/text_generation.html'
    )


@app.route('/zero_shot')
def zero_shot():
    return render_template(
        'generative_ai/zero_shot.html'
    )


# ==========================
# PREDICTIONS
# ==========================

#machine learning
#naive_bayes_predict
@app.route('/naive_bayes_predict', methods=['POST'])
def naive_bayes_predict():

    email = request.form['email']

    result = classify_email(email)

    return render_template(
        'result.html',
        title="Naive Bayes Result",
        result=result
    )

#knn_predict
@app.route('/knn_predict', methods=['POST'])
def knn_predict():

    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    result = predict_flower(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    return render_template(
        'result.html',
        title="KNN Prediction",
        result=result
    )    

#Linear Regression
@app.route('/linear_regression_predict', methods=['POST'])
def linear_regression_predict():

    file = request.files['dataset']

    filepath = os.path.join(
        "uploads",
        file.filename
    )

    file.save(filepath)

    result = run_linear_regression(filepath)

    return render_template(
        'result.html',
        title="Linear Regression",
        result=result,
        image="regression.png"
    )

#KMeans
@app.route('/kmeans_predict', methods=['POST'])
def kmeans_predict():

    file = request.files['dataset']

    filepath = os.path.join(
        "uploads",
        file.filename
    )

    file.save(filepath)

    run_kmeans(filepath)

    return render_template(
        'result.html',
        title="KMeans Clustering",
        result="KMeans completed successfully",
        image="kmeans.png"
    )

#DBSCAN
@app.route('/dbscan_predict', methods=['POST'])
def dbscan_predict():

    file = request.files['dataset']

    filepath = os.path.join(
        "uploads",
        file.filename
    )

    file.save(filepath)

    run_dbscan(filepath)

    return render_template(
        'result.html',
        title="DBSCAN Clustering",
        result="DBSCAN completed successfully",
        image="dbscan.png"
    )

#one_r
@app.route('/one_r_predict', methods=['POST'])
def one_r_predict():

    color = request.form['color']

    result = predict_one_r(color)

    return render_template(
        'result.html',
        title="One-R Prediction",
        result=result
    )

#deep learning
#ann
@app.route('/ann_predict', methods=['POST'])
def ann_predict():

    sepal_length = float(
        request.form['sepal_length']
    )

    sepal_width = float(
        request.form['sepal_width']
    )

    petal_length = float(
        request.form['petal_length']
    )

    petal_width = float(
        request.form['petal_width']
    )

    result = predict_flower_ann(

        sepal_length,
        sepal_width,
        petal_length,
        petal_width

    )

    return render_template(

        'result.html',

        title="ANN Prediction",

        result=result

    )

#cnn
@app.route('/cnn_predict', methods=['POST'])
def cnn_predict():

    file = request.files['image']

    filepath = os.path.join(

        "uploads",

        file.filename

    )

    file.save(filepath)

    result = predict_gender(filepath)

    return render_template(

        'result.html',

        title="CNN Prediction",

        result=result

    )    

#lstm
@app.route('/lstm_predict', methods=['POST'])
def lstm_predict():

    text = request.form['text']

    result = predict_sentiment(
        text
    )

    return render_template(

        'result.html',

        title="LSTM Prediction",

        result=result

    )

#generative_ai
#sentimental_analysis
@app.route('/sentiment_predict', methods=['POST'])
def sentiment_predict():

    text = request.form['text']

    result = analyze_sentiment(
        text
    )

    return render_template(
        'result.html',
        title="Sentiment Analysis",
        result=result
    )

 #Translation
@app.route('/translation_predict', methods=['POST'])
def translation_predict():

    text = request.form['text']

    result = translate_text(
        text
    )

    return render_template(

        'result.html',

        title="Translation",

        result=result

    )

#speech
@app.route('/speech_predict', methods=['POST'])
def speech_predict():

    file = request.files['audio']

    filepath = os.path.join(
        "uploads",
        file.filename
    )

    file.save(filepath)

    result = analyze_voice(filepath)

    return render_template(
        'result.html',
        title="Speech Analysis",
        result=result
    )

#qa
@app.route('/qa_predict', methods=['POST'])
def qa_predict():

    context = request.form['context']

    file = request.files['audio']

    filepath = os.path.join(
        "uploads",
        file.filename
    )

    file.save(filepath)

    result = answer_question(
        filepath,
        context
    )

    return render_template(
        'result.html',
        title="Voice Question Answering",
        result=result
    )

#ner
@app.route('/ner_predict', methods=['POST'])
def ner_predict():

    text = request.form['text']

    result = extract_entities(
        text
    )

    return render_template(

        'result.html',

        title="Named Entity Recognition",

        result=result

    )

#text generation
@app.route('/text_generation_predict', methods=['POST'])
def text_generation_predict():

    prompt = request.form['prompt']

    result = generate_text(prompt)

    return render_template(
        'result.html',
        title="Generated Text",
        result=result
    )    

#zero shot
@app.route('/zero_shot_predict', methods=['POST'])
def zero_shot_predict():

    text = request.form['text']

    result = classify_text(
        text
    )

    return render_template(
        'result.html',
        title="Zero Shot Learning",
        result=result
    )

#ARM
#apriori
@app.route('/apriori_predict', methods=['POST'])
def apriori_predict():

    file = request.files['dataset']

    filepath = os.path.join(
        "uploads",
        file.filename
    )

    file.save(filepath)

    result = run_apriori(
        filepath
    )

    return render_template(
        "result.html",
        title="Apriori Rules",
        result=result
    )
# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    app.run(debug=True)
