import numpy as np
import logging, os

UNK = "$UNK$"
NUM = "0"

def get_class(value):
	if value < 0.4:
		return 'nimp'
	elif value < 0.6:
		return 'imp'
	
	return 'vimp' 

def get_vocab(sents_ann):
	vocab_word, vocab_label = set(), set()
	for sent in sents_ann:
		for word, label in sent:
			if word.isdigit():
				word = '0'
			vocab_word.add(word.lower())
			vocab_label.add(label)
	return vocab_word, vocab_label



def get_senna_vocab(filename):
	vocab = set()
	with open(filename) as f:
		for line in f:
			word = line.strip()
			vocab.add(word)
		return vocab


def write_vocab(vocab, filename):
	with open(filename, "w") as f:
		for i, word in enumerate(vocab):
			if i != len(vocab) - 1:
				f.write("{}\n".format(word))
			else:
				f.write(word)


def load_vocab(filename):
    d_ = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            #word id actually start from 1
            d_[word] = idx

    return d_

def save_senna_vectors(vocab, vocab_emb, senna_filename, cmp_filename, dim):
    embeddings = np.zeros([len(vocab), dim])
    vocab_emb = list(vocab_emb)
    with open(senna_filename) as f:
        for i, line in enumerate(f):
            line = line.strip().split(' ')
            word = vocab_emb[i]
            embedding = map(float, line)
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(cmp_filename, embeddings=embeddings)


def get_word_vectors(filename):
    with open(filename) as f:
        return np.load(f)["embeddings"]

def clean_sent(sent):
	word_lst = []
	for word in sent.split():
		if len(word.strip()) > 0:
			if word.isalpha() or len(word) > 1:
				#remove trailing symbols
				if not word[-1].isalpha():
					word = word[:-1]
				word_lst.append(word)
	return word_lst


def get_logger(name):
	logger = logging.getLogger('logger')
	logger.setLevel(logging.DEBUG)
	logging.basicConfig(format='%(message)s', level=logging.DEBUG)

	handler = logging.FileHandler(name)
	handler.setLevel(logging.DEBUG)
	handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
	logging.getLogger().addHandler(handler)

	return logger

def create_folders(path):
	if not os.path.exists(path):
		os.makedirs(path)
	return path


def minibatches(data, batch_size):
	x_batch, y_batch, z_batch = [], [], []
	for (x, y, z) in data:
		if len(x_batch) == batch_size:
			yield x_batch, y_batch, z_batch
			x_batch, y_batch, z_batch = [], [], []

		x_batch += [x]
		y_batch += [y]	
		z_batch += [z]

	#this makes things tricky later
	if len(x_batch) != 0:
		yield x_batch, y_batch, z_batch


def pad_sequences(sequences, pad_tok):
	max_length = max(map(lambda x: len(x), sequences))
	sequence_padded, sequence_length = [], []

	for sequence in sequences:
		seq = list(sequence) + [pad_tok] * max(max_length - len(sequence), 0)
		sequence_padded += [seq]
		sequence_length += [min(len(sequence), max_length)]

	return sequence_padded, sequence_length

			
