DROP_INDICES = [
	"comment_code_value",
	"group_code_value",
	"parent_code_value"
]
SIGNIFICANT_INDICES = [
	"comment_text",
	"database_name",
	"letter_num",
]
ALL_INDICES = [
	"comment_text",
	"significance",
	"original_text",
	"database_name",
	"letter_num",
]
SKIP_FILES = [
	"public_comments_letters_20220708.csv",
]
VOCAB_SIZE = 100000
LETTER_MAP_INDICES = [
	"database_name",
	"letter_num",
	"letter_text"
]
TEXT_REPLACEMENT = {
	"{comma}" : ",",
	"{cr}{lf}" : "\r\n",
}