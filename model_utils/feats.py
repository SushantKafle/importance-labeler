import spacy
from spacy.tokens import Doc
from Entropy import get_entropy
import numpy as np


class WhitespaceTokenizer(object):
	def __init__(self, nlp):
		self.vocab = nlp.vocab

	def __call__(self, text):
		words = text.split(' ')
		spaces = [True] * len(words)
		return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en', create_make_doc=WhitespaceTokenizer)

POS_CODE = {
	'ADJ':0,
	'ADP': 1,
	'ADV': 2,	
	'AUX': 3,
	'CONJ': 4,	
	'DET': 5,	
	'INTJ': 6,
	'NOUN': 7,	
	'NUM': 8,	
	'PART': 9,	
	'PRON': 10,	
	'PROPN': 11,	
	'PUNCT': 12,	
	'SCONJ': 13,	
	'SYM': 14,	
	'VERB': 15,	
	'CCONJ': 16,
	'X': 17
	}

LIST_OF_CONTENT_POS = ["ADJ", "ADP", "ADV", "NOUN", "AUX"]

#given an index (i), provides left and right context on a list
def get_left_right_context(reference_words, i):
	left_context = reference_words[:i]
	right_context = reference_words[i+1:]
	return ' '.join(left_context), ' '.join(right_context)


#produces a big .csv files with feaures
def get_feats_file(files, outfile):
	file_obj = open(outfile, 'w')
	for i, f in enumerate(files):
		for (words, _, labels) in f:
			nwords = len(words)
			lengths = list(map(lambda x: len(x), words))
			min_length, max_length = min(lengths), max(lengths)
			sent = nlp(unicode(' '.join(words)))
			for i, word in enumerate(sent):
				feat = get_word_feature(sent, i, min_length, max_length, nwords)
				col = sum(feat, []) + [labels[i]]
				col = map(str, col)
				file_obj.write(','.join(col) + '\n')

		print ("file %d complete!" % (i + 1))

	file_obj.close()
	return outfile


#lookup features
def lookup_feats(feats_tbl, word_ids, sent_length, start_id, pad=True):
	features = []
	#print ("Lookup index: %d" % start_id)
	for index, batch in enumerate(word_ids):
		#print ("sub-index: %d" % index)
		feats = []
		feats[:] = feats_tbl[start_id + index]
		if pad:
			#print ("Sentence length (%d) vs Batch length (%d)" % (sent_length, len(feats)))
			for i in range(sent_length - len(feats)):
				feats.append([0] * 32)
		features.append(feats)
	return features


#given a minibatch (of word_ids), get features
def get_feats(words, idx_to_word, pad=True):
	features = []
	for batch in words:
		batch_ = [x for x in batch if x != -1]
		batch_words = [idx_to_word[idx] for idx in batch_]
		feats = get_feature(batch_words)
		if pad:
			for i in range(len(batch) - len(batch_)):
				feats.append([0] * 32)
				
		features.append(feats)
	return features

#given a list of words (i.e. sentence), get feaures
def get_feature(words):
	feats = []
	nwords = len(words)
	lengths = map(lambda x: len(x), words)
	min_length, max_length = min(lengths), max(lengths)
	sent = nlp(unicode(' '.join(words)))
	for i, word in enumerate(sent):
		feat = get_word_feature(sent, i, max_length, min_length, nwords)
		feats.append(sum(feat, []))

	return feats


#---compute feaures---#
def get_word_length(word, min_length, max_length):
	return (len(word.text) - min_length)/float(max_length - min_length) if max_length != min_length else len(word.text)/ (max_length)

def get_word_feature(sent, i, max_length, min_length, nwords):

	word = sent[i]
	prev_word = None if i == 0 else sent[i-1]
	next_word = None if i >= (nwords - 1) else sent[i+1]

	#length
	word_length_norm = get_word_length(word, min_length, max_length)
	next_word_length_norm = get_word_length(next_word, min_length, max_length) if next_word else 0
	prev_word_length_norm = get_word_length(prev_word, min_length, max_length) if prev_word else 0

	#positional features
	word_position_fwd = i/float(nwords - 1) if nwords != 1 else 1
	word_position_bwd = (nwords - i - 1)/float(nwords - 1) if nwords != 1 else 1

	#semantic features
	pos_tag = get_pos(word.pos_)
	content_function = 1 if word.pos_ in LIST_OF_CONTENT_POS else 0

	words = [w.text.lower() for w in sent]
	left_context, right_context = get_left_right_context(words, i)
	norm_entropy = get_entropy(left_context, right_context)

	if prev_word:
		left_context, right_context = get_left_right_context(words, i-1)
		prev_norm_entropy = get_entropy(left_context, right_context)
	else:
		prev_norm_entropy = 0

	if next_word:
		left_context, right_context = get_left_right_context(words, i+1)
		next_norm_entropy = get_entropy(left_context, right_context)
	else:
		next_norm_entropy = 0

	#dependency features
	num_children = len(list(word.children))/float(nwords)
	arc_distance_prev = 0 if i == 0 else get_arc_dist(word, sent[i - 1])/float(nwords)
	arc_distance_next = 0 if i >= (nwords - 1) else get_arc_dist(word, sent[i + 1])/float(nwords)
	dist_to_father = get_dist(word, word.head)/float(nwords)
	dist_to_gfather = get_dist(word, word.head.head)/float(nwords)

	feat = [
	#length based features (3 - 0, 1, 2)
	[word_length_norm, prev_word_length_norm, next_word_length_norm], 

	#positional features (2 - 3, 4)
	[word_position_fwd, word_position_bwd],

	#semantic features (5 - 5-22, 23, 24, 25, 26)
	pos_tag,
	[content_function, norm_entropy, prev_norm_entropy, next_norm_entropy],

	#dependency features (5 - 27, 28, 29, 30, 31)
	[num_children],
	[arc_distance_prev],
	[arc_distance_next],
	[dist_to_father],
	[dist_to_gfather]]

	#total features (15 | 32)
	return feat

def get_arc_dist(w1, w2):

	if w1 == w2:
		return 0
	if has_relation(w1, w2):
		return 1

	w1_ = w1
	cost_1 = 1
	while(True):
		w1_ = w1_.head
		cost_1 += 1
		if has_relation(w1_, w2):
			break

		if (w1_.head == w1_):
			break

	w2_ = w2
	cost_2 = 1
	while(True):
		w2_ = w2_.head
		cost_2 += 1
		if has_relation(w1, w2_):
			break

		if (w2_.head == w2_):
			break

	return min(cost_1, cost_2)


def has_relation(w1, w2):
	return w1.head == w2 or w2.head == w1


def get_dist(from_, to_):
	return abs(from_.i - to_.i)


def get_word(ids, word_map):
	return map(lambda x: word_map[x], ids)


def get_pos(pos_tag):
	tags = [0] * len(POS_CODE)
	id_ = POS_CODE[pos_tag]
	tags[id_] = 1
	return tags