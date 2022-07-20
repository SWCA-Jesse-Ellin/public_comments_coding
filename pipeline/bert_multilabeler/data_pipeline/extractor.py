from pipeline.data_pipeline.extractor import Extractor

import pandas as pd
import os

class BERTExtractor(Extractor):
	def load(self, coded_path, letter_path, coded_indices=[], letter_indices):
		coded_df = pd.DataFrame()
		letter_df = pd.DataFrame()
		for file in os.listdir(coded_path):
			local_df = pd.read_csv(file)
			coded_df = coded_df.combine(local_df, overwrite=False)
		for file in os.listdir(coded_path):
			local_df = pd.read_csv(file)
			letter_df = letter_df.combine(local_df)
		self.coded_data = coded_df.copy()
		self.letter_data = letter_df.copy()

	def dump(self):
		return {"codes" : self.coded_data.copy(), "letters" : self.letter_data.copy()}