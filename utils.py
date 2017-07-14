import numpy as np


def get_class(value):
	if value < 0.4:
		return 'nimp'
	elif value < 0.6:
		return 'limp'
	elif value < 0.8:
		return 'imp'
	
	return 'vimp' 

def get_vocab(sents_ann):
	vocab_word, vocab_label = set(), set()
	for sent in sents_ann:
		for word, label in sent:
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
			
