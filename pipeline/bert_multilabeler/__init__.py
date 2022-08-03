from pipeline.bert_multilabeler.data_pipeline.pipe import BERTDataPipeline

import os
import pandas as pd
import spacy

CODE_PATH = "test_data/raw/code_data"
LETTER_PATH = "test_data/raw/letter_mappings"
ETL_PATH = "test_data/etl_output/bert"
OUTPUT_FILE = "bert_parsed_comments.csv"

def generateData(sep='\n'):
	sp = spacy.load('en_core_web_sm')
	stopwords = sp.Defaults.stop_words

	if not os.path.isdir(ETL_PATH):
		os.mkdir(ETL_PATH)

	data_pipe = BERTDataPipeline(save_dir=ETL_PATH, stopwords=stopwords)
	data_pipe.parse(CODE_PATH, LETTER_PATH, OUTPUT_FILE, sep=sep)

def getData(generate=False, sep='\n'):
	if generate:
		generateData(sep)
	return pd.read_csv(f"{ETL_PATH}/{OUTPUT_FILE}")