import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
import numpy as np

# Last inn modellen
model = tf.keras.models.load_model('imdb_sentiment_model.h5')

# Funksjon for å forutsi sentimentet til nye anmeldelser
def predict_sentiment(text):
    # Parametere
    vocab_size = 18000
    max_length = 5000
    trunc_type = 'post'
    padding_type = 'post'

    # Last ned IMDb ordindeksen
    word_index = imdb.get_word_index()
    
    # Juster ordindeksen til modellens ordforråd
    adjusted_word_index = {word: (index + 3) if (index + 3) < vocab_size else 2 for word, index in word_index.items()}

    # Konverter ordene i teksten til sekvenser av indekser
    sequences = [[adjusted_word_index.get(word, 2) for word in text.lower().split()]]
    
    # Padde sekvensene
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    
    # Forutsi sentimentet
    prediction = model.predict(padded)
    return prediction[0][0]

# Lese anmeldelsen fra en tekstfil
with open("input.txt", "r", encoding="utf-8") as file:
    new_review = file.read()

# Skriv ut og forutsi sentimentet
goodornot = predict_sentiment(new_review)
if goodornot > 0.5:
    result = 'positive'
else:
    result = 'negative'
print("Review:", new_review)
print("Is the text positive or negative? \nThe text is:", result)
