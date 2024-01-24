import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
import numpy as np

# Parameters (these can be adjusted)
vocab_size = 18000
embedding_dim = 16
max_length = 5000
trunc_type = 'post'
padding_type = 'post'
num_epochs = 6  # Adjust this for training
dropout_rate = 0.01  # Dropout rate

# Load IMDb dataset
(train_data, train_labels), (_, _) = imdb.load_data(num_words=vocab_size)

# Preprocess data
train_data = pad_sequences(train_data, maxlen=max_length, padding=padding_type, truncating=trunc_type)
train_labels = np.array(train_labels)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

# Build the model with dropout layers
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    GlobalAveragePooling1D(),
    Dropout(dropout_rate),  # Dropout layer after pooling
    Dense(24, activation='relu'),
    Dropout(dropout_rate),  # Another dropout layer after the first Dense layer
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=num_epochs, validation_data=(X_val, y_val), verbose=2)

# Save the model
model.save('imdb_sentiment_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val, verbose=2)
print(f'Validation Accuracy: {accuracy}')
