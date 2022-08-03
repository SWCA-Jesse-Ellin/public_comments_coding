from pipeline.data_pipeline.extractor import Extractor

import pandas as pd
import os

class BERTExtractor(Extractor):
	def load(self, coded_path, letter_path):
		coded_df = None
		letter_df = None
		for file in os.listdir(coded_path):
			local_df = pd.read_csv(f"{coded_path}/{file}")
			if type(coded_df) == type(None):
				coded_df = local_df.copy()
			else:
				coded_df = pd.concat([coded_df, local_df])
		for file in os.listdir(letter_path):
			local_df = pd.read_csv(f"{letter_path}/{file}")
			if type(letter_df) == type(None):
				letter_df = local_df.copy()
			else:
				letter_df = pd.concat([letter_df, local_df])
		self.coded_data = coded_df.copy()
		self.letter_data = letter_df.copy()

	def dump(self):
		return {"codes" : self.coded_data.copy(), "letters" : self.letter_data.copy()}