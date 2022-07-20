import tensorflow as tf
from tensorflow import keras

from pipeline.binary_classifier.constants import VOCAB_SIZE
from pipeline.binary_classifier.model_pipeline.model import BinaryModel

class LSTMModel(BinaryModel):
	def generate_model(self, loss="custom"):
		if loss=="custom":
			loss = super(LSTMModel, self).custom_loss()
		self.model = keras.Sequential()
		self.model.add(keras.layers.Embedding(VOCAB_SIZE, 128))
		self.model.add(keras.layers.Dropout(rate=0.1))
		self.model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)))
		self.model.add(keras.layers.Bidirectional(keras.layers.LSTM(64)))
		self.model.add(keras.layers.Dropout(rate=0.1))
		self.model.add(keras.layers.Dense(1, activation='sigmoid'))
		print(self.model.summary())

		self.model.compile(optimizer='adam', loss=loss, metrics=['accuracy', tf.keras.metrics.Recall(name="recall")])