from config import config
import os, random, csv
from nltk.tokenize import sent_tokenize
from utils import *
from tokenizer import TedliumTokenizer


#read and tokenize the scripts to sentences
print ("Reading the script..")
allowable_ext = ['txt', 'stm']
sentences = {}
ted_tokenizer = TedliumTokenizer()

for script_file in os.listdir(config.script_path):
	file_ext = script_file[-3:]
	all_sentences = []
	
	if not file_ext in allowable_ext:
		continue

	with open(os.path.join(config.script_path, script_file)) as script_:
		if file_ext == 'txt':
			all_sentences = sent_tokenize(script_.read())
			sentences[script_file] = all_sentences
		elif file_ext == 'stm':
			all_sentences = ted_tokenizer.tokenize(script_)
			sentences[script_file] = all_sentences

print ("Total files: %d" % len(sentences))


#read the annotation file
print ("Reading the annotations..")
annotations = {}
for script_file in sentences.keys():
	all_annotations = []
	#change extension to csv file
	annotation_file = script_file + '.csv'
	print (annotation_file)
	with open(os.path.join(config.script_path, annotation_file)) as annotation_:
		csv_reader = csv.reader(annotation_)
		next(csv_reader, None)
		for row in csv_reader:
			all_annotations.append(get_class(float(row[1])))
	annotations[script_file] = all_annotations
		
#attaching the importance score to the sentence
all_sentences_ann = []
for script_file in sentences.keys():
	idx = 0
	annotations_ = annotations[script_file]
	for sent in sentences[script_file]:
		words = clean_sent(sent)
		if len(words) != 0:
			all_sentences_ann.append(zip(words, annotations_[idx: idx + len(words)]))
			idx += len(words)

#building vocab
print ("Building vocab..")
vocab_words, vocab_tags = get_vocab(all_sentences_ann)
vocab_emb = get_senna_vocab(os.path.join(config.senna_home, 'hash/words.lst'))
vocab_words = vocab_words & vocab_emb
vocab_words.add(UNK)
vocab_words.add(NUM)

print ("Num words: %d" % len(vocab_words))
print ("Num tags: %d" % len(vocab_tags))

#saving vocab
print ("Saving vocab..")
write_vocab(vocab_words, config.words_vocab_path)
write_vocab(vocab_tags, config.tags_vocab_path)

#loading vocab
print ("Loading vocab..")
vocab_words = load_vocab(config.words_vocab_path)
vocab_tags = load_vocab(config.tags_vocab_path)

#saving word embeddings for the vocabulary
print ("Preparing embeddings for the vocab..")
save_senna_vectors(vocab_words, vocab_emb, os.path.join(config.senna_home, 'embeddings/embeddings.txt'), 
	config.cmp_embeddings_src, config.word_embedding_dim)

#creating the training and test files
random.shuffle(all_sentences_ann)

#80% of the data is used for training and remaning for test and dev
train_hinge = int(len(all_sentences_ann) * 0.8)
train_ = all_sentences_ann[:train_hinge + 1]	
test_ = all_sentences_ann[train_hinge + 1:]


#writing the train and the test data
print ("Preparing training and test dataset..")
with open(config.train_data, 'w') as fobj:
	for sent in train_:
		for word, label in sent:
			fobj.write('%s\t%s\n' % (word, label))
		fobj.write('\n')

with open(config.test_data, 'w') as fobj:
	for sent in test_:
		for word, label in sent:
			fobj.write('%s\t%s\n' % (word, label))

		fobj.write('\n')


print ("Done building the data!")

