# Initialize Libs
from tables import *
import math, os, untangle, operator
import numpy as np
from scipy import spatial

SWITCHBOARD_H5FILE_SRC = '/Users/sxk5664/Dropbox/Research/REU\'17/code/ngrams/new-swd-callhome-ted-ngrams.h5'
h5filelocation = SWITCHBOARD_H5FILE_SRC
print ("Working with h5 file: %s." % h5filelocation)
assert os.path.isfile(h5filelocation), "File Doesn't exist!"

#open file in read mode
h5file = open_file(h5filelocation, 'r')
print (h5file)

onegrams = h5file.root.onegrams
twograms = h5file.root.bigrams
threegrams = h5file.root.trigrams
fourgrams = h5file.root.fourgrams
fivegrams = h5file.root.fivegrams

NUM_CANDIDATES = 10
eps = 0.1**10

def count(ngram, token):
    query = ' & '.join(['(w%d == b"%s")' % ((i+1), w) for i, w in enumerate(token)])
    cnt = 0
    for res in ngram.where(query):
        cnt += res['freq']
    return cnt

#returns top candidates given the context
def get_candidates(context, left=True):
    candidates = {}
    tokens = context.split()
    num_candidates = 0
    
    if left:
        #eligible for four gram 
        if len(tokens) >= 4 and fivegrams != None:
            predictions = {}
            fourgram = tokens[-4:]
            for fg in fivegrams.where('(w1 == b"%s") & (w2 == b"%s") & (w3 == b"%s") & (w4 == b"%s")' % tuple(fourgram)):
                if fg['w5'] in predictions:
                    predictions[fg['w5']][0] += fg['freq']
                else:
                    predictions[fg['w5']] = [fg['freq'], count(fourgrams, fourgram)]
                    
                num_candidates += 1
                #if num_candidates > NUM_CANDIDATES:
                #    break

            candidates['n5'] = predictions

        if len(tokens) >= 3 and fourgrams != None:# and num_candidates < NUM_CANDIDATES:
            predictions = {}
            threegram = tokens[-3:]
            for fg in fourgrams.where('(w1 == b"%s") & (w2 == b"%s") & (w3 == b"%s")' % tuple(threegram)):
                if fg['w4'] in predictions:
                    predictions[fg['w4']][0] += fg['freq']
                else:
                    predictions[fg['w4']] = [fg['freq'], count(threegrams, threegram)]
                
                num_candidates += 1
                #if num_candidates > NUM_CANDIDATES:
                #    break

            candidates['n4'] = predictions

        if len(tokens) >= 2:# and num_candidates < NUM_CANDIDATES:
            predictions = {}
            twogram = tokens[-2:]
            for tg in threegrams.where('(w1 == b"%s") & (w2 == b"%s")' % tuple(twogram)):
                if tg['w3'] in predictions:
                    predictions[tg['w3']][0] += tg['freq']
                else:
                    predictions[tg['w3']] = [tg['freq'], count(twograms, twogram)]
                
                num_candidates += 1
                #if num_candidates > NUM_CANDIDATES:
                #    break
                    
            candidates['n3'] = predictions

        if len(tokens) >= 1:# and num_candidates < NUM_CANDIDATES:
            predictions = {}
            onegram = tokens[-1]
            for tg in twograms.where('w1 == b"%s"' % onegram):
                if tg['w2'] in predictions:
                    predictions[tg['w2']][0] += tg['freq']
                else:
                    predictions[tg['w2']] = [tg['freq'], count(onegrams, [onegram])]
                    
                num_candidates += 1
                #if num_candidates > NUM_CANDIDATES:
                #    break

            candidates['n2'] = predictions

        if num_candidates < NUM_CANDIDATES:
            predictions = {}
            for i in range(NUM_CANDIDATES - num_candidates):
                predictions[str(i)] = [0.1,10000]
            candidates['n1'] = predictions
    else:
        #eligible for four gram 
        if len(tokens) >= 4:
            predictions = {}
            fourgram = tokens[:4]
            for fg in fivegrams.where('(w2 == b"%s") & (w3 == b"%s") & (w4 == b"%s") & (w5 == b"%s")' % tuple(fourgram)):
                if fg['w1'] in predictions:
                    predictions[fg['w1']][0] += fg['freq']
                else:
                    predictions[fg['w1']] = [fg['freq'], count(fourgrams, fourgram)]
                    
                num_candidates += 1
                #if num_candidates > NUM_CANDIDATES:
                #    break

            candidates['n5'] = predictions

        if len(tokens) >= 3:# and num_candidates < NUM_CANDIDATES:
            predictions = {}
            threegram = tokens[:3]
            for fg in fourgrams.where('(w2 == b"%s") & (w3 == b"%s") & (w4 == b"%s")' % tuple(threegram)):
                if fg['w1'] in predictions:
                    predictions[fg['w1']][0] += fg['freq']
                else:
                    predictions[fg['w1']] = [fg['freq'], count(threegrams, threegram)]
                    
                num_candidates += 1
                #if num_candidates > NUM_CANDIDATES:
                #    break

            candidates['n4'] = predictions

        if len(tokens) >= 2:# and num_candidates < NUM_CANDIDATES:
            predictions = {}
            twogram = tokens[:2]
            for tg in threegrams.where('(w2 == b"%s") & (w3 == b"%s")' % tuple(twogram)):
                if tg['w1'] in predictions:
                    predictions[tg['w1']][0] += tg['freq']
                else:
                    predictions[tg['w1']] = [tg['freq'], count(twograms, twogram)]
                    
                num_candidates += 1
                #if num_candidates > NUM_CANDIDATES:
                #    break

            candidates['n3'] = predictions

        if len(tokens) >= 1:# and num_candidates < NUM_CANDIDATES:
            predictions = {}
            onegram = tokens[0]
            for tg in twograms.where('w2 == b"%s"' % onegram):
                if tg['w1'] in predictions:
                    predictions[tg['w1']][0] += tg['freq']
                else:
                    predictions[tg['w1']] = [tg['freq'], count(onegrams, [onegram])]
                    
                num_candidates += 1
                #if num_candidates > NUM_CANDIDATES:
                #    break

            candidates['n2'] = predictions

        if num_candidates < NUM_CANDIDATES:
            predictions = {}
            for i in range(NUM_CANDIDATES - num_candidates):
                predictions[str(i)] = [0.1,10000]
            candidates['n1'] = predictions
    #print (candidates)     
    return candidates

def stupidbackoff_scoring(candidates):
    scored_cand = []
    penalty = {'n5': 1, 'n4': 0.4, 'n3': 0.4**2, 'n2': 0.4**3, 'n1': 0.4**4}
    for ngram, predictions in candidates.items():
        for word, count in predictions.items():
                score = penalty[ngram] * (count[0]/float(count[1]))
                scored_cand.append([word, score])
    
    return sorted(scored_cand, key=lambda x: x[1], reverse=True)
                
def get_probability(candidates):
    t_sum = sum([x for w,x in candidates])
    return [[w, x/float(t_sum) if x != 0 else eps] for w, x  in candidates]

def combine(l_candidates, r_candidates):
    result = dict((w,i) for w, i in l_candidates)
    penalty = 0.4
    for w, i in r_candidates:
        if w in result:
            result[w] += i
        else:
            result[w] = penalty * i
    return sorted(result.items(), key=operator.itemgetter(1), reverse=True)

def truncate_and_normalize(candidates):
    t_candidates = {}
    max_, min_ = 0, 10
    for w, score in candidates:
        
        min_ = score if score < min_ else min_
        max_ = score if score > max_ else max_
        
        t_candidates[w] = score
        
        if len(t_candidates) == NUM_CANDIDATES:
            break
            
    if max_ == min_:
        return [[w, x] for w, x in t_candidates.items()]
    
    return [[w, x] for w, x in t_candidates.items()]

#returns the entropy given the context
def get_entropy(left_context, right_context, display= False):
    #using left context to get the candidates the their probability
    
    #candidates predicted using the left context
    #each candidate will have the word, the count
    #and n-gram
    l_candidates = stupidbackoff_scoring(get_candidates(left_context))
    r_candidates = stupidbackoff_scoring(get_candidates(right_context, left=False))
    
    candidates = truncate_and_normalize(combine(l_candidates, r_candidates))
    
    n_candidates = get_probability(candidates)
    if display:
        print ('Top Candidates:', n_candidates)
        
    entropy = (-1 * sum([x * math.log(x,2) for w, x in n_candidates]))/float(math.log(len(n_candidates), 2))
    return entropy




