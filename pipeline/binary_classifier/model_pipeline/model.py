import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras.backend as K

from pipeline.binary_classifier.constants import VOCAB_SIZE

class BinaryModel():
	def __init__(self, load_file=None, loss="custom"):
		if not load_file:
			self.generate_model(loss)
		else:
			self.model = keras.models.load_model(load_file)

	def generate_model(self, loss="custom"):
		if loss=="custom":
			loss = self.custom_loss()
		self.model = keras.Sequential()
		self.model.add(keras.layers.Embedding(VOCAB_SIZE, 16))
		self.model.add(keras.layers.Conv1D(128, 7, padding='valid', activation="relu", strides=3))
		self.model.add(keras.layers.GlobalAveragePooling1D())
		self.model.add(keras.layers.Dense(16, activation='relu'))
		self.model.add(keras.layers.Dropout(rate=0.1))
		self.model.add(keras.layers.Dense(1, activation='sigmoid'))
		print(self.model.summary())

		self.model.compile(optimizer='adam', loss=self.custom_loss(), metrics=['accuracy', tf.keras.metrics.Recall(name="recall")], run_eagerly=True)

	def custom_loss(self):
		def f1(y_true, y_pred):
			y_pred = K.round(y_pred)
			tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
			tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
			fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
			fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

			p = tp / (tp + fp + K.epsilon())
			r = tp / (tp + fn + K.epsilon())

			f1 = 2*p*r / (p+r+K.epsilon())
			f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
			return K.mean(f1)

		def f1_loss(y_true, y_pred):
			y_true = tf.where(y_true, 1., 0.)
			tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
			tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
			fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
			fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

			p = tp / (tp + fp + K.epsilon())
			r = tp / (tp + fn + K.epsilon())

			f1 = 2*p*r / (p+r+K.epsilon())
			f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
			return 1 - K.mean(f1)

		return f1_loss

	def train(self, x_train, y_train, epochs, batch_size, val_data, test=None, callbacks=None):
		history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=val_data, callbacks=callbacks)
		if test:
			print(f"Post-Train Accuracy: {self.model.evaluate(test[0], test[1])}")
		return history

	def evaluate(self, x, y):
		return self.model.evaluate(x, y)
		
	def to_binary(self, Y, threshold=0.5):
		return np.array([True if y[0] >= threshold else False for y in Y])

	def predict(self, x, threshold=0.5):
		return self.to_binary(self.model.predict(x), threshold=threshold)

	def save(self, filepath):
		return self.model.save(filepath)