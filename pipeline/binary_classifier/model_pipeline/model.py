import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from pipeline.binary_classifier.constants import VOCAB_SIZE

class BinaryModel():
	def __init__(self):
		self.generate_model()

	def generate_model(self):
		self.model = keras.Sequential()
		self.model.add(keras.layers.Embedding(VOCAB_SIZE, 16))
		self.model.add(keras.layers.GlobalAveragePooling1D())
		self.model.add(keras.layers.Dense(16, activation='relu'))
		self.model.add(keras.layers.Dense(1, activation='sigmoid'))
		print(self.model.summary())

		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	def train(self, x_train, y_train, epochs, batch_size, val_data):
		history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=val_data)

	def predict(self, x):
		return self.model.predict(x)