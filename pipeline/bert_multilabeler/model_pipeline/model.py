import torch
device = torch.device("cpu")
if torch.cuda.is_available():
	device = torch.device("cuda")
from transformers import BertForPretraining, BertTokenizer

from pipeline.binary_classifier.constants import VOCAB_SIZE
from pipeline.bert_multilabeler.model_pipeline.bert_master import tokenization

class BERTMultilabeler():
	def __init__(self, n_classes):
		self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		self.bert = BertForPretraining.from_pretrained("bert-base-uncased")
		self.drop = torch.nn.Dropout(p=0.3)
		self.lstm = torch.nn.LSTM(input_size=)

	def generateModel(self):
		pass

	def train(self, x_train, y_train, epochs, batch_size, val_data, test_data=None, callbacks=None):
		x_train = self.tokenizer(x_train, return_tensors="pt")
		val_data[0] = self.tokenizer(val_data[0], return_tensors="pt")
		if test_data:
			test_data[0] = self.tokenizer(test_data[0], return_tensors="pt")
		pass

	def evaluate(self, x, y):
		pass

	def predict(self, x):
		pass

	def save(self, filepath):
		pass