import pandas as pd

class Extractor():
	def __init__(self, basedir=""):
		self.parent_directory = basedir
		self.data = None

	def load(self, filepath, indices=[]):
		if filepath.split('.')[-1] == "csv":
			self.data = pd.read_csv(filepath, encoding_errors='ignore')
		else:
			self.data = pd.read_excel(filepath)
      
	def dump(self):
		return self.data.copy()