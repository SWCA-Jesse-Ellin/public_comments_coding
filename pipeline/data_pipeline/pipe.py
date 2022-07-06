from extractor import Extractor
from transformer import Transformer
from loader import Loader

import pandas as pd

class DataPipeline():
	def __init__(self, read_dir="", save_dir="", stopwords):
		self.extractor = Extractor(read_dir)
		self.transformer = Transformer(stopwords)
		self.loader = Loader(save_dir)

	def parse(self, filepaths, target_filepath):
		data = pd.DataFrame()
		comments = []
		codes = []
		for filepath in filepahts:
			self.extractor.load(filepath)
			local_data = self.transformer.transform(self.extractor.dump())
			comments += local_data["comment_text"]
			codes += local_data["comment_code_value"]
		data["comment_text"] = comments
		data["comment_code_value"] = comments
		self.loader.save(data, target_filepath)