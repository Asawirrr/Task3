import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources (Run this once)
nltk.download('stopwords')
nltk.download('wordnet')


# Load dataset
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

df_fake["label"] = 0  # Fake News = 0
df_real["label"] = 1  # Real News = 1

df = pd.concat([df_fake, df_real], axis=0)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle dataset

print("Dataset Loaded Successfully!")
df.head()



# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

df['text'] = df['title'] + ' ' + df['text']  # Combine title and text
df['text'] = df['text'].apply(preprocess_text) # ek ek row ko bhejega aur jo return hoga unn preprocessed words sey row ko replace krega... for loop ki trah chal rha ha lekin for loop sey efficient ha .apply

print("Text Preprocessing Completed!")
df.head()



from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization & Padding
MAX_NB_WORDS = 50000  # Max words in vocabulary
MAX_SEQUENCE_LENGTH = 500  # Max length of input sequence
EMBEDDING_DIM = 100  # Embedding size

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

y = df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Tokenization & Padding Completed!")
print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

from tensorflow.keras.models import Sequential
# Define LSTM Model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=1)

print("LSTM Model Training Completed!")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'LSTM Accuracy: {accuracy * 100:.2f}%')
from sklearn.metrics import classification_report, roc_auc_score

# Generate a classification report (Precision, Recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Calculate AUC-ROC Score
auc_roc = roc_auc_score(y_test, y_pred)
print(f'AUC-ROC Score: {auc_roc:.2f}')
import pickle

# Save the trained model
model.save("lstm_model.keras")

# Save the tokenizer
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle)

print("Model and Tokenizer Saved Successfully!")