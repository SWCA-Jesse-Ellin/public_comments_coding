import re

class Transformer():
	def __init__(self, stopwords=[]):
		self.stopwords = stopwords

	def transform(self, data):
		return data.copy