from pipeline.data_pipeline.pipe import DataPipeline
from pipeline.model_pipeline.NaiveModel import NaiveInterface

import os
import pandas as pd
import spacy
sp = spacy.load('en_core_web_sm')
stopwords = sp.Defaults.stop_words

INPUT_PATH = "test_data/raw"
ETL_PATH = "test_data/etl_output"
OUTPUT_FILE = "parsed_comments.csv"
if not os.path.isdir(ETL_PATH):
	os.mkdir(ETL_PATH)

data_pipe = DataPipeline(read_dir=INPUT_PATH, save_dir=ETL_PATH, stopwords=stopwords)
data_pipe.parse(os.listdir(INPUT_PATH), OUTPUT_FILE)

data = pd.read_csv(f"{ETL_PATH}/{OUTPUT_FILE}")

model = NaiveInterface()
print(model.predict(data["comment_text"], Y=data["comment_code_value"]))

print(data.head())