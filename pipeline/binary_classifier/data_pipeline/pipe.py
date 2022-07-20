from pipeline.data_pipeline.pipe import DataPipeline
from pipeline.binary_classifier.data_pipeline.extractor import BinaryExtractor
from pipeline.binary_classifier.data_pipeline.transformer import BinaryTransformer
from pipeline.binary_classifier.data_pipeline.loader import BinaryLoader
from pipeline.binary_classifier.constants import SIGNIFICANT_INDICES, ALL_INDICES, LETTER_MAP_INDICES

import pandas as pd

class BinaryDataPipeline(DataPipeline):
	def __init__(self, read_dir="", save_dir="", stopwords=[]):
		self.extractor = BinaryExtractor(read_dir)
		self.transformer = BinaryTransformer(stopwords)
		self.loader = BinaryLoader(save_dir)

	def parse(self, filepaths, blank_paths, target_filepath, filetype=None, skip=[], sep='\n'):
		data = pd.DataFrame(columns=ALL_INDICES)
		for filepath in filepaths:
			if filetype and filepath.split('.')[1] != filetype:
				continue
			if filepath.split('/')[-1] in skip:
				continue
			self.extractor.load(filepath, indices=SIGNIFICANT_INDICES)
			local_data = self.transformer.transform(self.extractor.dump())
			data = pd.concat([data, local_data])
		blank_df = pd.DataFrame(columns=LETTER_MAP_INDICES)
		for filepath in blank_paths:
			local_df = self.extractor.extract_blank(filepath)
			blank_df = pd.concat([blank_df, local_df])
		blank_data = self.transformer.pull_blanks(blank_df, data, sep=sep)
		self.loader.save(blank_data, target_filepath)