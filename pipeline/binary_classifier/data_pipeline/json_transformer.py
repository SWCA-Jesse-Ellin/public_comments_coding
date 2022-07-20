from pipeline.data_pipeline.transformer import Transformer

import re
import pandas as pd
import json
from tqdm import tqdm

class JSONBinaryTransformer(Transformer):
	def __init__(self, stopwords=[]):
		super(JSONBinaryTransformer, self).__init__(stopwords)

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

	def reorganize(self, json_data):
		data = {}
		json_data = json_data["RECORDS"]
		for entry in json_data:
			local_data = json.loads(entry['letters_json'])
			if local_data["database_name"] not in data.keys():
				data[local_data["database_name"]] = dict()
			data[local_data["database_name"]][local_data["letter_num"]] = [entry.strip() for entry in local_data["letter_text"].split('.') if entry != '']
		return data

	def paragraph_pull(self, blank_data, labeled_data):
		blank_data = self.reorganize(blank_data)

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
			while j < len(json_data[db][letter]):
				# t.set_description(f"DB: {db}, Letter: {letter}, p: {j}")
				if original_text in json_data[db][letter][j]:
					json_data[db][letter].pop(j)
					j -= 1
				if len(json_data[db][letter]) <= 0:
					continue
				j += 1

		for i in tqdm(range(len(texts))):
			original_text = o_text[i]
			db = dbs[i]
			letter = nums[i]

			doc_text = json_data[db][letter]
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
		data = {}
		blank_data = blank_data["RECORDS"]
		for entry in blank_data:
			local_data = json.loads(entry['letters_json'])
			if local_data["database_name"] not in data.keys():
				data[local_data["database_name"]] = dict()
			data[local_data["database_name"]][local_data["letter_num"]] = local_data["letter_text"]
		blank_data = data

		texts = labeled_data["comment_text"].tolist()
		text_set = set(texts)
		sig = labeled_data["significance"].tolist()
		print(f"Started with {len(sig)} true data points")
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
				parsed_text = self.process_text(entry).strip().lower()
				if parsed_text in text_set:
					continue
				o_text.append(entry)
				texts.append(parsed_text)
				sig.append(False)
				dbs.append(db)
				nums.append(letter)

		return self.removeBlanks(pd.DataFrame({
					"comment_text" : texts,
					"significance" : sig,
					"original_text" : o_text,
					"database_name" : dbs,
					"letter_num" : nums
				})).sort_values("significance").drop_duplicates(subset="comment_text", keep="last")

	def pull_blanks(self, json_data, labeled_data, sep):
		if sep == "\n":
			return self.paragraph_pull(json_data, labeled_data)
		if sep == ".":
			return self.sentence_pull(json_data, labeled_data)

	def transformJSON(self, json_data):
		dbs = []
		nums = []
		com_codes = []
		com_code_vals = []
		par_codes = []
		par_code_vals = []
		com_text = []

		for line in json_data["RECORDS"]:
			line = json.loads(line["comments_json"])
			dbs.append(line["database_name"])
			nums.append(line["letter_num"])
			com_codes.append(line["comment_code"])
			com_code_vals.append(line["comment_code_value"])
			par_codes.append(line["parent_code"])
			par_code_vals.append(line["parent_code_value"])
			com_text.append(line["comment_text"])

		return pd.DataFrame({
			"database_name" : dbs,
			"letter_num" : nums,
			"comment_code" : com_codes,
			"comment_code_value" : com_code_vals,
			"parent_code" : par_codes,
			"parent_code_value" : par_code_vals,
			"comment_text" : com_text
		})
