DROP_INDICES = [
	"letter_num",
	"database_name",
	"comment_code",
	"comment_code_value",
	"parent_code_value",
	"letter_text",
	"original_text"
]
INSIGNIFICANT_CODE = "~~~~"
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
BERT_MODEL_DIR = "pipeline/bert_multilabeler/model_pipeline/bert_uncased_L-12_H-768_A-12"
BERT_INIT_CKPNT = "bert_model.ckpt.data-00000-of-00001"
BERT_VOCAB = "vocab.txt"