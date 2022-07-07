from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print("Running data pipeline...")
from pipeline.binary_classifier import data
from pipeline.binary_classifier.model_pipeline.model import BinaryModel

print("Processing data...")
x = data["comment_text"]
y = data["significance"]

tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(x)
x_tokens = tokenizer.texts_to_sequences(x)
word_index = tokenizer.word_index
# set key index values
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

x_pad = pad_sequences(x_tokens, value=word_index["<PAD>"], padding="post", maxlen=256)

x_train, x_test, y_train, y_test = train_test_split(x_pad, y, test_size=1/20, random_state=0)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1/50, random_state=0)

print("Generating model...")
model = BinaryModel()
model.train(x_train, y_train, 40, 1024, (x_val, y_val))
