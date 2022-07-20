from pipeline.data_pipeline.extractor import Extractor
import json
import pandas as pd

class BinaryExtractor(Extractor):
	def load(self, filepath, indices=[]):
		super(BinaryExtractor, self).load(filepath)
		if indices:
			self.data = self.data[indices]

	def extract_json(self, filepath):
		with open(filepath, 'r') as f:
			return json.load(f)

	def extract_blank(self, filepath):
		return pd.read_csv(filepath)