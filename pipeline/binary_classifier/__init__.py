from pipeline.binary_classifier.data_pipeline.pipe import BinaryDataPipeline
# from pipeline.binary_classifier.data_pipeline.json_pipe import JSONBinaryDataPipeline
from pipeline.binary_classifier.constants import SKIP_FILES

import os
import pandas as pd
import spacy

INPUT_PATH = "test_data/raw/code_data"
LETTER_MAP_PATH = "test_data/raw/letter_mappings"
ETL_PATH = "test_data/etl_output"
OUTPUT_FILE = "parsed_comments.csv"
	
def generateData(sep='\n'):
	sp = spacy.load('en_core_web_sm')
	stopwords = sp.Defaults.stop_words

	if not os.path.isdir(ETL_PATH):
		os.mkdir(ETL_PATH)

	data_pipe = BinaryDataPipeline(read_dir=INPUT_PATH, save_dir=ETL_PATH, stopwords=stopwords)
	data_pipe.parse([f"{INPUT_PATH}/{file}" for file in os.listdir(INPUT_PATH)],
					[f"{LETTER_MAP_PATH}/{file}" for file in os.listdir(LETTER_MAP_PATH)],
					OUTPUT_FILE,
					filetype='csv',
					skip=SKIP_FILES,
					sep=sep)

def getData(generate=False, sep='\n'):
	if generate:
		generateData(sep)
	return pd.read_csv(f"{ETL_PATH}/{OUTPUT_FILE}")