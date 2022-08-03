from pipeline.data_pipeline.loader import Loader

class BERTLoader(Loader):
	def __init__(self, basedir=""):
		super(BERTLoader, self).__init__(basedir)

	def save(self, data, filepath):
		super(BERTLoader, self).save(data, filepath)