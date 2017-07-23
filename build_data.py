from config import config
import os, random, csv
from nltk.tokenize import sent_tokenize
from utils import *


#read and tokenize the scripts to sentences
print ("Reading the script..")
with open(config.script_src) as script_:
	all_sentences = sent_tokenize(script_.read())
print ("Total sentences: %d" % len(all_sentences))


#read the annotation file
print ("Reading the annotations..")
annotations = []
with open(config.annotation_src) as annotation_:
	csv_reader = csv.reader(annotation_)
	next(csv_reader, None)
	for row in csv_reader:
		annotations.append(get_class(float(row[1])))
		
#attaching the importance score to the sentence
idx = 0
all_sentences_ann = []
for sent in all_sentences:
	words = clean_sent(sent)
	if len(words) != 0:
		all_sentences_ann.append(zip(words, annotations[idx: idx + len(words)]))
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

