class Loader():
	def __init__(self, basedir=""):
		self.parent_directory = basedir

	def save(self, data, filepath):
		data.to_csv(f"{self.parent_directory}/{filepath}", index=False)