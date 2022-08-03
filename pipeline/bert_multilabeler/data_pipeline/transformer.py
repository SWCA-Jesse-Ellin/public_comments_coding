from pipeline.data_pipeline.transformer import Transformer
from pipeline.binary_classifier.constants import TEXT_REPLACEMENT, PUNCTUATION_SPLITTERS
from pipeline.bert_multilabeler.constants import INSIGNIFICANT_CODE

import re
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.semi_supervised import LabelSpreading

class BERTTransformer(Transformer):
	def __init__(self, stopwords=[], sep='.', spread=False):
		super(BERTTransformer, self).__init__(stopwords)
		self.sep = sep
		self.spread = spread

	def removeBlanks(self, data):
		filter = data["comment_text"] != ''
		data = data[filter]
		return data.dropna()

	def process_text(self, text):
		text = re.sub("[^a-zA-Z0-9]", " ", text)
		text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
		text = re.sub(r"\s+", " ", text)
		text = " ".join([word for word in text.split(' ') if word not in self.stopwords])
		return text.strip().lower()

	def preprocess(self, data):
		new_text = []
		old_text = data["comment_text"]
		for sentence in old_text:
			new_text.append(self.process_text(sentence))
		data["comment_text"] = new_text
		data["original_text"] = old_text
		return data

	def spreadLabels(self, df, f=BAD_LABEL_LAMBDA):
		df = df.copy()
		labels = df["parent_code"].tolist()
		labels = [item if not re.search(r'[a-z0-9.\s]', item) else '' for item in labels]
		df["spread_codes"] = labels
		spreader = LabelSpreading()

	def transform(self, data):
		data = data.copy()
		data = self.removeBlanks(data)
		data = self.preprocess(data)
		data = self.removeBlanks(data)
		if self.spread:
			return self.spreadLabels(data)
		return data

	def restructureBlanks(self, blank_data):
		new_data = dict()
		dbs, nums, texts = blank_data["database_name"].tolist(), blank_data["letter_num"].tolist(), blank_data["letter_text"].tolist()
		t = tqdm(range(len(dbs)))
		for i in t:
			if dbs[i] not in new_data.keys():
				new_data[dbs[i]] = dict()
			new_data[dbs[i]][nums[i]] = texts[i]

		return new_data

	def paragraphPull(self, coded_data, letter_data, kwargs):
		t = tqdm(range(len(kwargs["texts"])))
		for i in t:
			original_text = kwargs["o_text"][i]
			db = kwargs["dbs"][i]
			letter = kwargs["nums"][i]
			letter_data[db][letter] = letter_data[db][letter].split(self.sep)
			j = 0
			while j < len(letter_data[db][letter]):
				if original_text in letter_data[db][letter][j]:
					letter_data[db][letter].pop(j)
					j-=1
				if len(letter_data[db][letter]) <= 0:
					continue
				j+=1

		for db in tqdm(letter_data.keys()):
			for letter in tqdm(letter_data[db].keys(), leave=False):
				doc_text = letter_data[db][letter]
				for entry in doc_text:
					kwargs["o_text"].append(entry)
					kwargs["texts"].append(self.process_text(entry))
					kwargs["codes"].append(INSIGNIFICANT_CODE)
					kwargs["dbs"].append(db)
					kwargs["nums"].append(letter)

		return self.removeBlanks(pd.DataFrame({
				"comment_text" : kwargs["texts"],
				"parent_code" : kwargs["codes"],
				"original_text" : kwargs["o_text"],
				"database_name" : kwargs["dbs"],
				"letter_num" : kwargs["nums"]
			})).drop_duplicates(subset="comment_text", keep="first")

	def sentencePull(self, coded_data, letter_data, kwargs):
		for i in tqdm(range(len(kwargs["texts"]))):
			original_text = kwargs["o_text"][i]
			db = kwargs["dbs"][i]
			letter = kwargs["nums"][i]

			doc_text = [entry.strip() for entry in re.split(PUNCTUATION_SPLITTERS, letter_data[db][letter])]
			for entry in doc_text:
				if original_text in entry:
					continue
				kwargs["o_text"].append(entry)
				kwargs["texts"].append(self.process_text(entry))
				kwargs["codes"].append(INSIGNIFICANT_CODE)
				kwargs["dbs"].append(db)
				kwargs["nums"].append(letter)

		return self.removeBlanks(pd.DataFrame({
				"comment_text" : kwargs["texts"],
				"parent_code" : kwargs["codes"],
				"original_text" : kwargs["o_text"],
				"database_name" : kwargs["dbs"],
				"letter_num" : kwargs["nums"]
			})).drop_duplicates(subset="comment_text", keep="first")

	def replaceText(self, text):
		for k,v in TEXT_REPLACEMENT.items():
			text = text.replace(k,v)
		return text

	def assignBlanks(self, coded_data, letter_data):
		letter_data["letter_text"] = letter_data["letter_text"].map(self.replaceText)
		letter_data = self.restructureBlanks(letter_data)
		kwargs = {
			"texts" : coded_data["comment_text"].tolist(),
			"codes" : coded_data["parent_code"].tolist(),
			"o_text" : coded_data["original_text"].tolist(),
			"dbs" : coded_data["database_name"].tolist(),
			"nums" : coded_data["letter_num"].tolist()
		}
		if self.sep == '.':
			return self.sentencePull(coded_data, letter_data, kwargs)
		if self.sep == '\n':
			return self.paragraphPull(coded_data, letter_data, kwargs)

	def encode(self, data):
		encoder = LabelBinarizer()
		encoder.fit(data["parent_code"].unique())
		encoded = encoder.transform(data["parent_code"].to_numpy().reshape(-1,1))
		cols = sorted([(k,v) for k,v in encoder.get_params().items()], key=lambda x:x[0])
		ohe_df = pd.DataFrame(encoded, columns=encoder.classes_)
		return pd.concat([data, ohe_df], axis=1).drop(["parent_code"], axis=1)
