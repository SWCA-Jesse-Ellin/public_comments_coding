from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
from time import time
from tqdm import tqdm
import datetime
import os

print("Running data pipeline...")
from pipeline.binary_classifier import getData
data = getData(generate=False, sep='.')
from pipeline.binary_classifier.model_pipeline.model import BinaryModel
from pipeline.binary_classifier.model_pipeline.lstm import LSTMModel
from pipeline.binary_classifier.constants import VOCAB_SIZE

print("Processing data...")
print(data["significance"].value_counts())
print(f"False ratio: {len(data[data['significance'] == False]['significance'].tolist()) / len(data['significance'].tolist()):.2%}")
###
# This block sets balanced training data
###
# train_sigs = data[data["significance"] == True]
# insigs = data[data["significance"] == False]
# print(f"Significant data count: {len(train_sigs['significance'].tolist())}\nInsiginificant data count: {len(insigs['significance'].tolist())}")
# num_sig = len(train_sigs["significance"].tolist()) // 2
# train_insig = insigs.sample(n=num_sig)
# training = pd.concat([train_sigs,train_insig])
x = data["comment_text"]
y = data["significance"]

# tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
# tokenizer.fit_on_texts(x)
# tokenizer_json = tokenizer.to_json()
# with open(f"saved_models/tokenizer_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json", 'w') as f:
# 	f.write(tokenizer_json)
tokenizer_save_time = "20220719-132636"
with open(f"saved_models/tokenizer_{tokenizer_save_time}.json") as f:
	tokenizer = tokenizer_from_json(f.read())
x_tokens = tokenizer.texts_to_sequences(x)
word_index = tokenizer.word_index
# set key index values
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

x_pad = pad_sequences(x_tokens, value=word_index["<PAD>"], padding="post", maxlen=256)

x_train, x_test, y_train, y_test = train_test_split(x_pad, y, test_size=1/20, random_state=0, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/10, random_state=0, stratify=y_train)

def evaluate(models, x_train, y_train, epochs, batch_sizes, val_data, test_data, save=False):
	df = pd.DataFrame(columns=["model", "loss", "training_time", "training_accuracy", "training_recall", "validation_accuracy", "validation_recall", "testing_accuracy", "testing_recall"])
	t = tqdm(models.items())
	i = 0
	for name, model in t:
		batch_size = batch_sizes[name.split('_')[0]]
		model.generate_model(loss)
		log_dir = f"logs/fit/{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
		early_stop = tf.keras.callbacks.EarlyStopping(
			monitor="val_recall",
			min_delta=0.005,
			patience=3,
			verbose=1,
			mode='max',
			baseline=None,
			restore_best_weights=True
		)
		t.set_description(f"Running {name}...")
		start = time()
		history = model.train(x_train, y_train,
							  epochs, batch_size,
							  val_data,
							  callbacks=[tensorboard_callback, early_stop])
		end = time()
		y_pred = model.predict(test_data[0])
		m = tf.keras.metrics.Recall()
		m.update_state(y_pred, test_data[1])
		test_recall = m.result().numpy()
		y_pred = model.predict(x_train)
		m.update_state(y_pred, y_train)
		train_recall = m.result().numpy()
		y_pred = model.predict(val_data[0])
		m.update_state(y_pred, val_data[1])
		val_recall = m.result().numpy()
		test_acc = model.evaluate(test_data[0], test_data[1])[1]
		df.loc[len(df.index)] = [
			name,
			loss,
			f"{end-start:.2f}",
			history.history['accuracy'][-1],
			train_recall,
			history.history['val_accuracy'][-1],
			val_recall,
			test_acc,
			test_recall
		]
		if save:
			model.save(f"saved_models/{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
	return df

EPOCHS = 10000
BATCH_SIZES = {
	"BinaryModel" : 2048,
	"LSTM" : 1024
}
losses = [
	"custom",
	"binary_crossentropy"
]

models = dict()
for loss in losses:
	models[f"BinaryModel_{loss}"] = BinaryModel(loss=loss)
	models[f"LSTM_{loss}"] = LSTMModel(loss=loss)
results = evaluate(models, x_train, y_train, EPOCHS, BATCH_SIZES, val_data=(x_val, y_val), test_data=(x_test, y_test), save=True)

print(results)
file_name = "_".join([name for name in models.keys()]).lower()
results.to_csv(f"model_results/{file_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
###
# This block loads models from memory
###
# models = dict()
# binary_timestamp = ""
# lstm_timestamp = ""
# save_dir = "saved/models"
# for model in os.listdir(save_dir):
#	if os.path.isdir(model):
#		model_name = model.split('_')[0]
#		if model_name == "BinaryModel" and model.split('_')[-1]==binary_timestamp:
#			models[model.split('_')[0:2]] = BinaryModel(load_file=model)
#		elif model_name == "LSTM" and model.split('_')[-1]==lstm_timestamp:
#			models[model.split('_')[0:2]] = LSTMModel(load_file=model))
# binary_model_timestamp = "20220719-132710"
# lstm_timestamp = "20220719-134218"
# models = {
# 	"BinaryModel" : BinaryModel(load_file=f"saved_models/BinaryModel_{binary_model_timestamp}"),
# 	"LSTM" : LSTMModel(load_file=f"saved_models/LSTM_{lstm_timestamp}")
# }
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
for (name, model), ax in zip(models.items(), axs.flatten()):
	y_pred = model.predict(x_test, threshold=0.8)
	cm = confusion_matrix(y_pred, y_test, normalize="true")
	sns.heatmap(cm, annot=True, cmap="Blues", ax=ax, xticklabels=[False, True], yticklabels=[False, True])
	ax.title.set_text(name)
	ax.set_ylabel("Actual")
	ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"test_confusion_matrix_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
plt.show()
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
for (name, model), ax in zip(models.items(), axs.flatten()):
	y_pred = model.predict(x_pad, threshold=0.8)
	cm = confusion_matrix(y_pred, y, normalize="true")
	sns.heatmap(cm, annot=True, cmap="Blues", ax=ax, xticklabels=[False, True], yticklabels=[False, True])
	ax.title.set_text(name)
	ax.set_ylabel("Actual")
	ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig(f"model_results/test_confusion_matrix_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
plt.show()