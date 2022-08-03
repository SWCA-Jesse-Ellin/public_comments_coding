from pipeline.data_pipeline.pipe import DataPipeline
from pipeline.bert_multilabeler.data_pipeline.extractor import BERTExtractor
from pipeline.bert_multilabeler.data_pipeline.transformer import BERTTransformer
from pipeline.bert_multilabeler.data_pipeline.loader import BERTLoader

import pandas as pd

class BERTDataPipeline(DataPipeline):
	def __init__(self, read_dir="", save_dir="", stopwords=[]):
		self.extractor = BERTExtractor(read_dir)
		self.transformer = BERTTransformer(stopwords)
		self.loader = BERTLoader(save_dir)

	def parse(self, coded_path, letter_path, target_path, sep='\n'):
		self.extractor.load(coded_path, letter_path)
		data = self.extractor.dump()
		coded_data = data["codes"]
		letter_data = data["letters"]
		coded_data = self.transformer.transform(coded_data)
		data = self.transformer.assignBlanks(coded_data, letter_data)
		self.loader.save(data, target_path)