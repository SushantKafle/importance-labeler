import spacy
from spacy.tokens import Doc

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


class WhitespaceTokenizer(object):
	def __init__(self, nlp):
		self.vocab = nlp.vocab

	def __call__(self, text):
		words = text.split(' ')
		# All tokens 'own' a subsequent space character in this tokenizer
		spaces = [True] * len(words)
		return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en', create_make_doc=WhitespaceTokenizer)

def get_feats(words):
	features = []
	for batch in words:
		feats = []
		batch_ = [x for x in batch if type(x) == str]
		nwords = len(batch_)
		lengths = map(lambda x: len(x), batch_)
		min_length, max_length = min(lengths), max(lengths)
		sent = nlp(unicode(' '.join(batch_)))
		for i, word in enumerate(sent):
			word_length_norm = (len(word.text) - min_length)/float(max_length - min_length) if max_length != min_length else len(word.text)/ (max_length)
			word_position_fwd = i/float(nwords - 1) if nwords != 1 else 1
			word_position_bwd = (nwords - i - 1)/float(nwords - 1) if nwords != 1 else 1
			pos_tag = get_pos(word.pos_)
			is_capitalized = int(word.text[0].isupper())

			num_children = len(list(word.children))/float(nwords)
			arc_distance_prev = 0 if i == 0 else get_arc_dist(word, sent[i - 1])/float(nwords)
			arc_distance_next = 0 if i >= (nwords - 1) else get_arc_dist(word, sent[i + 1])/float(nwords)
			dist_to_father = get_dist(word, word.head)/float(nwords)
			dist_to_gfather = get_dist(word, word.head.head)/float(nwords)

			feat = [[word_length_norm], 
				[word_position_fwd],
				[word_position_bwd],
				pos_tag,
				[is_capitalized],
				[num_children],
				[arc_distance_prev],
				[arc_distance_next],
				[dist_to_father],
				[dist_to_father]]
			feats.append(sum(feat, []))

		for i in range(len(batch) - len(batch_)):
			feats.append([0] * 27)
		features.append(feats)
	return features


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