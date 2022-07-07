import pandas as pd

class Extractor():
	def __init__(self, basedir=""):
		self.parent_directory = basedir
		self.data = None

	def load(self, filepath):
		if filepath.split('.')[-1] == "csv":
			self.data = pd.read_csv(f"{self.parent_directory}/{filepath}")
		else:
			self.data = pd.read_excel(f"{self.parent_directory}/{filepath}")

	def dump(self):
		return self.data.copy()