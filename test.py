from model.feats import *
from align_text import align_text

def test_get_arc_dist_test():
	sent = nlp(u'I am so disappointed in you.')
	am = sent[1]
	so = sent[2]
	dist = get_arc_dist(am, so)
	print (dist)
	assert(dist == 2)

def test_get_features_test():
	print (get_feature(['i', 'am', 'so', 'disappointed', 'with', 'you']))


def test_alignment():
	text_1 = "i have a meeting on thursday but i don't think i will be able to make it"
	text_2 = "have a meeting on thursday don't think well be able bake eat"
	alignments, error_info = align_text(text_1, text_2, display=True)

	assert (error_info['I'] == 0)
	assert (error_info['S'] == 3)


test_alignment()





