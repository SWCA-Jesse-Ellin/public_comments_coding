from pipeline.data_pipeline.loader import Loader

class BinaryLoader(Loader):
	def __init__(self, basedir=""):
		super(BinaryLoader, self).__init__(basedir)

	def save(self, data, filepath):
		super(BinaryLoader, self).save(data, filepath)