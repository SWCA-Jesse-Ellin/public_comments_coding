from pipeline.data_pipeline.pipe import DataPipeline
from pipeline.binary_classifier.data_pipeline.extractor import BinaryExtractor
from pipeline.binary_classifier.data_pipeline.json_transformer import JSONBinaryTransformer
from pipeline.binary_classifier.data_pipeline.loader import BinaryLoader
from pipeline.binary_classifier.constants import SIGNIFICANT_INDICES, ALL_INDICES

import pandas as pd

class JSONBinaryDataPipeline(DataPipeline):
	def __init__(self, read_dir="", save_dir="", stopwords=[]):
		self.extractor = BinaryExtractor(read_dir)
		self.transformer = JSONBinaryTransformer(stopwords)
		self.loader = BinaryLoader(save_dir)

	def parse(self, filepaths, json_path, target_filepath, filetype=None, skip=[]):
		data = pd.DataFrame(columns=ALL_INDICES)
		for filepath in filepaths:
			datum = self.extractor.extract_json(filepath)
			local_data = self.transformer.transformJSON(datum)
			data = pd.concat([data, self.transformer.transform(local_data)[ALL_INDICES]])
		blank_df = self.extractor.extract_json(json_path[0])
		blank_data = self.transformer.pull_blanks(blank_df, data, sep='.')
		self.loader.save(blank_data, target_filepath)