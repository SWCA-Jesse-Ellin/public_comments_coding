import re

class Transformer():
	def __init__(self, stopwords=[]):
		self.stopwords = stopwords

	def removeBlanks(self, data):
		filter = data["comment_text"] != ""
		data = data[filter]
		filter = data["comment_code_value"] != ""
		data = data[filter]
		return data.dropna()

	def preprocess(self, data):
		new_text = []
		old_text = data["comment_text"]
		for sentence in old_text:
			# remove punctuation, single characters, and double spaces
			sentence = re.sub('[^a-zA-Z0-9]', ' ', sentence)
			sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
			sentence = re.sub(r'\s+', ' ', sentence)
			sentence = " ".join([word for word in sentence.split(' ') if word not in self.stopwords])
			new_text.append(sentence)
		data["comment_text"] = new_text
		return data

	def transform(self, data):
		data = data.copy()
		data = self.removeBlanks(data)
		data = self.preprocess(data)

		return data