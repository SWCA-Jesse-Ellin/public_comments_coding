from pipeline.data_pipeline.transformer import Transformer

import re
from random import sample # REMOVE FOR DEPLOYMENT

class BinaryTransformer(Transformer):
	def __init__(self, stopwords=[]):
		super(BinaryTransformer, self).__init__(stopwords)

	def removeBlanks(self, data):
		filter = data["comment_text"] != ""
		data = data[filter]
		return data.dropna()

	def preprocess(self, data):
		if "group_code" not in data.columns:
			data["group_code"] = data["parent_code"]
		new_text = []
		old_text = data["comment_text"]
		arbitration = sample(data["group_code"].unique().tolist(), len(data["group_code"].unique().tolist()) // 2) # REMOVE FOR DEPLOYMENT
		for sentence in old_text:
			# remove punctuation, single characters, and double spaces
			sentence = re.sub('[^a-zA-Z0-9]', ' ', sentence)
			sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
			sentence = re.sub(r'\s+', ' ', sentence)
			sentence = " ".join([word for word in sentence.split(' ') if word not in self.stopwords])
			new_text.append(sentence.strip().lower())
		data["comment_text"] = new_text
		data["original_text"] = old_text
		# data["significance"] = [True for _ in old_text] # UNCOMMENT FOR DEPLOYMENT
		data["significance"] = [True if data["group_code"].iloc[i] not in arbitration else False for i in range(len(old_text))] # REMOVE FOR DEPLOYMENT
		return data

	def transform(self, data):
		data = data.copy()
		data = self.removeBlanks(data)
		data = self.preprocess(data)

		return data