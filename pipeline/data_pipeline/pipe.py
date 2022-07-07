from pipeline.data_pipeline.extractor import Extractor
from pipeline.data_pipeline.transformer import Transformer
from pipeline.data_pipeline.loader import Loader
from pipeline.data_pipeline.constants import SIGNIFICANT_INDICES

import pandas as pd

class DataPipeline():
	def __init__(self, extractor=None, transformer=None, loader=None, read_dir="", save_dir="", stopwords=[]):
		self.extractor = Extractor(read_dir) if not extractor else extractor
		self.transformer = Transformer(stopwords) if not transformer else transformer
		self.loader = Loader(save_dir) if not loader else loader

	def parse(self, filepaths, target_filepath):
		data = pd.DataFrame(columns=SIGNIFICANT_INDICES)
		for filepath in filepaths:
			self.extractor.load(filepath)
			local_data = self.transformer.transform(self.extractor.dump())
			data = pd.concat([data, local_data])
		self.loader.save(data, target_filepath)