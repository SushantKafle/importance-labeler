from docx import Document
from nltk.tokenize import sent_tokenize


def get_text(filename):
	doc = Document(filename)
	full_text = []
	for para in doc.paragraphs:
		full_text.append(para.text)

	return '\n'.join(full_text)


def read_watson_meta(filename):
	words = []
	full_text = get_text(filename)
	headers = full_text.split('\n')[0].strip()

	if len(headers.split('\t')) != 4:
		return

	for line in full_text.split('\n')[1:]:
		if line.strip() != '':
			word = line.split()[0]
			if word.strip() != "%HESITATION":
				words.append(word)
	return str(' '.join(words))


def get_sents(filename):
	text = get_text(filename)
	sents = sent_tokenize(text)
	if len(sents) > 0:
		return sents

def clean_texts(texts):
	if type(texts) == str:
		texts = [texts]

	clean_texts = []
	for text in texts:
		#lower case
		text = text.lower()

		#remove symbols like commas, dash etc.
		words = []
		for word in text.split():
			if not word[-1].isalnum():
				word = word[:-1]

			if word.strip() == '' or (len(word) == 1 and not word.isalnum):
				continue
			
			words.append(word)

		clean_texts.append(' '.join(words))
	return clean_texts





