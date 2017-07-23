from feats import *

def get_arc_dist_test():
	sent = nlp(u'I am so disappointed in you.')
	am = sent[1]
	so = sent[2]
	dist = get_arc_dist(am, so)
	print (dist)
	assert(dist == 2)

def get_features_test():
	print get_feats([['I', 'am', 'so', 'disappointed', 'with', 'you']])

get_features_test()
