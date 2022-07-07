from pipeline.data_pipeline.extractor import Extractor

class BinaryExtractor(Extractor):
	def load(self, filepath, indices=[]):
		super(BinaryExtractor, self).load(filepath)
		if indices:
			self.data = self.data[indices]