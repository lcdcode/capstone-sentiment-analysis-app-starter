import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
app = Flask(__name__)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("user_text")
    else:
        text = ""
    sentiment = SentimentIntensityAnalyzer()
    sentiment_score = sentiment.polarity_scores(text)
    sentiment = sentiment.polarity_scores(text) # VADER results
    # create a new key in the dictionary to store the custom model sentiment analysis results
    load_tokenizer()
    load_keras_model()
    sentiment["custom model positive"] = sentiment_analysis(text)
    return render_template('form.html', sentiment=sentiment)

model = None
tokenizer = None

def load_keras_model():
    global model
    model = load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

# @app.before_first_request
def before_first_request():
    load_keras_model()
    load_tokenizer()

def sentiment_analysis(input):
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]),2)

if __name__ == "__main__":
    app.run()
