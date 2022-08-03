import argparse
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
import spacy

from pipeline.binary_classifier.data_pipeline.pipe import BinaryDataPipeline
from pipeline.binary_classifier.model_pipeline.lstm import LSTMModel

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, help="Path to csv file that stores input data (must have \"letter_text\" as columns)")
args = parser.parse_args()

tokenizer_save_time = "20220719-132636"
with open(f"saved_models/tokenizer_{tokenizer_save_time}.json") as f:
	tokenizer = tokenizer_from_json(f.read())

sp = spacy.load('en_core_web_sm')
stopwords = sp.Defaults.stop_words
sep = '.'
pipeline = BinaryDataPipeline(stopwords=stopwords)
data = pipeline.extractor.load(args.input_file)
data = pipeline.transformer.pull_blanks(pipeline.extractor.dump(), labeled_data=pd.DataFrame({"comment_text":[], "significance":[], "original_text":[], "database_name":[], "letter_num":[]}), sep='.')

x = data["comment_text"]
word_index = tokenizer.word_index
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3
x_tokens = tokenizer.texts_to_sequences(x)
x_pad = pad_sequences(x_tokens, value=word_index["<PAD>"], padding="post", maxlen=256)

batch_size = 1024
model_timestamp = "20220719-165216"
model = LSTMModel(load_file=f"saved_models/LSTM_{model_timestamp}")

results = model.predict(x_pad)
binary = model.to_binary(results, threshold=0.5)

data["significance"] = binary
data["confidence"] = results

data.to_csv("binary_model_results.csv")