from pipeline.data_pipeline.pipe import DataPipeline
from pipeline.binary_classifier.data_pipeline.extractor import BinaryExtractor
from pipeline.binary_classifier.data_pipeline.transformer import BinaryTransformer
from pipeline.binary_classifier.data_pipeline.loader import BinaryLoader
from pipeline.binary_classifier.constants import SIGNIFICANT_INDICES, ALL_INDICES

import pandas as pd

class BinaryDataPipeline(DataPipeline):
	def __init__(self, read_dir="", save_dir="", stopwords=[]):
		self.extractor = BinaryExtractor(read_dir)
		self.transformer = BinaryTransformer(stopwords)
		self.loader = BinaryLoader(save_dir)

	def parse(self, filepaths, target_filepath):
		data = pd.DataFrame(columns=ALL_INDICES)
		for filepath in filepaths:
			self.extractor.load(filepath)#, indices=SIGNIFICANT_INDICES)
			local_data = self.transformer.transform(self.extractor.dump())
			data = pd.concat([data, local_data])
		self.loader.save(data, target_filepath)