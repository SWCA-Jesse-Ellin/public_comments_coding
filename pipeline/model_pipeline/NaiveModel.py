from interface import ModelInterface

class NaiveInterface(ModelInterface):
	def __init__(self):
		pass

	def train(self, X, Y, split_ratio):
		pass

	def predict(self, X, Y=None):
		return Y

	def analyze(self, Y_hat, Y_pred):
		pass