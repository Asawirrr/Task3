from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load LSTM model and tokenizer
model = tf.keras.models.load_model("lstm_model.keras")
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Constants
MAX_SEQUENCE_LENGTH = 500  # Same as training

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    text_sequence = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return text_padded

# Home page with form
@app.route('/')
def home():
    return render_template("index.html")  # Renders HTML page

# Handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get("news_text", "")
    
    if not news_text:
        return render_template("index.html", prediction="Please enter news text!")

    # Preprocess input and predict
    text_input = preprocess_text(news_text)
    prediction = model.predict(text_input)[0][0]  # Get probability
    result = "Real News" if prediction > 0.5 else "Fake News"

    return render_template("index.html", prediction=result, confidence=f"{prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
