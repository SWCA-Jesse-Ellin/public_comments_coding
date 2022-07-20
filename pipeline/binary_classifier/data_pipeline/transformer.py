from pipeline.data_pipeline.transformer import Transformer
from pipeline.binary_classifier.constants import TEXT_REPLACEMENT

import re
import pandas as pd
from tqdm import tqdm

class BinaryTransformer(Transformer):
	def __init__(self, stopwords=[]):
		super(BinaryTransformer, self).__init__(stopwords)

	def removeBlanks(self, data):
		filter = data["comment_text"] != ""
		data = data[filter]
		return data.dropna()

	def process_text(self, text):
		text = re.sub('[^a-zA-Z0-9]', ' ', text)
		sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
		text = re.sub(r'\s+', ' ', text)
		text = " ".join([word for word in text.split(' ') if word not in self.stopwords])
		return text

	def preprocess(self, data):
		new_text = []
		old_text = data["comment_text"]
		for sentence in old_text:
			new_text.append(self.process_text(sentence).strip().lower())
		data["comment_text"] = new_text
		data["original_text"] = old_text
		data["significance"] = [True for _ in old_text]
		return data

	def transform(self, data):
		data = data.copy()
		data = self.removeBlanks(data)
		data = self.preprocess(data)
		data = self.removeBlanks(data)

		return data

	def restructure_blanks(self, blank_data):
		new_data = dict()
		dbs, nums, texts = blank_data["database_name"].tolist(), blank_data["letter_num"].tolist(), blank_data["letter_text"].tolist()
		t = tqdm(range(len(dbs)))
		for i in t:
			if dbs[i] not in new_data.keys():
				new_data[dbs[i]] = dict()
			new_data[dbs[i]][nums[i]] = texts[i]

		return new_data

	def paragraph_pull(self, blank_data, labeled_data):
		texts = labeled_data["comment_text"].tolist()
		sig = labeled_data["significance"].tolist()
		o_text = labeled_data["original_text"].tolist()
		dbs = labeled_data["database_name"].tolist()
		nums = labeled_data["letter_num"].tolist()

		# preprocess paragraphs out of original text
		t = tqdm(range(len(texts)))
		for i in t:
			original_text = o_text[i]
			db = dbs[i]
			letter = nums[i]
			j = 0
			while j < len(blank_data[db][letter]):
				# t.set_description(f"DB: {db}, Letter: {letter}, p: {j}")
				if original_text in blank_data[db][letter][j]:
					blank_data[db][letter].pop(j)
					j -= 1
				if len(blank_data[db][letter]) <= 0:
					continue
				j += 1

		for db in tqdm(blank_data.keys()):
			for letter in tqdm(blank_data[db].keys(), leave=False):
				doc_text = blank_data[db][letter]
				for entry in doc_text:
					o_text.append(entry)
					texts.append(self.process_text(entry).strip().lower())
					sig.append(False)
					dbs.append(db)
					nums.append(letter)

		return self.removeBlanks(pd.DataFrame({
					"comment_text" : texts,
					"significance" : sig,
					"original_text" : o_text,
					"database_name" : dbs,
					"letter_num" : nums
				})).drop_duplicates(subset="comment_text", keep="first")

	def sentence_pull(self, blank_data, labeled_data):
		texts = labeled_data["comment_text"].tolist()
		sig = labeled_data["significance"].tolist()
		o_text = labeled_data["original_text"].tolist()
		dbs = labeled_data["database_name"].tolist()
		nums = labeled_data["letter_num"].tolist()

		for i in tqdm(range(len(texts))):
			original_text = o_text[i]
			db = dbs[i]
			letter = nums[i]

			blank_data[db][letter].replace(original_text, '')

		for i in tqdm(range(len(texts))):
			original_text = o_text[i]
			db = dbs[i]
			letter = nums[i]

			doc_text = [entry.strip() for entry in blank_data[db][letter].split('.')]
			for entry in doc_text:
				if original_text in entry:
					continue
				o_text.append(entry)
				texts.append(self.process_text(entry).strip().lower())
				sig.append(False)
				dbs.append(db)
				nums.append(letter)

		return self.removeBlanks(pd.DataFrame({
					"comment_text" : texts,
					"significance" : sig,
					"original_text" : o_text,
					"database_name" : dbs,
					"letter_num" : nums
				})).drop_duplicates(subset="comment_text", keep="first")

	def replace_text(self, text):
		for k,v in TEXT_REPLACEMENT.items():
			text = text.replace(k,v)
		return text

	def pull_blanks(self, blank_data, labeled_data, sep='\n'):
		blank_data["letter_text"] = blank_data["letter_text"].map(self.replace_text)
		blank_data = self.restructure_blanks(blank_data)

		if sep == '\n':
			return self.paragraph_pull(blank_data, labeled_data)

		if sep == '.':
			return self.sentence_pull(blank_data, labeled_data)
