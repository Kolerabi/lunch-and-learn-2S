import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
import numpy as np

# Parameters (these can be adjusted)
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 25000
num_epochs = 10  # Adjust this for training

# Load IMDb dataset
(train_data, train_labels), (_, _) = imdb.load_data(num_words=vocab_size, oov_char=oov_tok)

# Preprocess data
train_data = pad_sequences(train_data, maxlen=max_length, padding=padding_type, truncating=trunc_type)
train_labels = np.array(train_labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=2)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val, verbose=2)
print(f'Validation Accuracy: {accuracy}')