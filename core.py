from docx_utils import get_sents, read_watson_meta, get_text
import re, datetime, csv
from align_text import align_text
import time
from model.config import config
from collections import Counter

#contains a template for the HTML page
HTML_TEMPLATE = "data/template.html"


'''
Computes WER

input:
reference: reference text
hypothesis: hypothesis text
logger: logger

output:
returns a dict containing WER and other useful statistics
'''
def compute_wer(reference, hypothesis, logger):
	stats = {}
	logger.info("Computing WER (this may take awhile)..")
	alignments, error_cnt = align_text(reference, hypothesis)

	stats['num_errors'] = error_cnt['D'] + error_cnt['I'] + error_cnt['S']
	stats['num_substitution'] = error_cnt['S']
	stats['num_insertion'] = error_cnt['I']
	stats['num_deletion'] = error_cnt['D']

	stats['num_ref_words'] = len(reference.split())
	stats['num_hyp_words'] = len(hypothesis.split())
	logger.info("Total words in the reference text: %d" % stats['num_ref_words'])

	WER = (stats['num_errors'])/float(stats['num_ref_words'])

	stats['WER'] = WER
	stats['alignments'] = alignments
	logger.info("WER: %f" % WER)

	return stats



'''
Labeling reference text with word
importance information

input:
reference: reference text
process_word: function to process the word 
to adhere to model format
outfile: path for saving the output file
logger: logger

output:
returns a dict containing word importance 
information and other useful statistics
'''
def importance_labeling(model, reference_text, process_word, outfile, logger):
	logger.info("Annotating the reference text with word importance (this will take some time..)")
	labels = model.label_text(reference_text.split('\n'), process_word)

	logger.info("Saving the annotation information in a CSV file")
	words = reference_text.split( )
	with open(outfile, 'w') as f:
		writer = csv.writer(f)
		for word, label in zip(words, labels):
			writer.writerow([word, label])
	
	logger.info("CSV file saved at: %s" % outfile)
	counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
	counter = Counter(labels)
	for idx, count in counter.iteritems():
		counts[idx] = count
	return {'counts': counts, 'imp_label': labels}


'''
Computes WWER

input:
alignments: alignment information between the reference
and the hypothesized words
labels: word importance labels
N: number of reference words
logger: logger

output:
returns a dict containing WWER and other useful statistics
'''
def compute_wwer(alignments, labels, N, logger):
	logger.info("Computing WWER..")
	cost = 0
	error_imp_cmb = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
	for alignment in alignments:
		if alignment['H'] != alignment['R']:
			ref_id = alignment['ID']
			if ref_id == -1:
				#insertion error
				len_hypothesis = len(alignment['H'])
				cost += 0.05 * len_hypothesis
			else:
				imp_score = labels[ref_id]
				error_imp_cmb[imp_score] += 1
				cost += (imp_score)

	WWER = cost/float(N)
	logger.info("WWER = %f" % WWER)
	return {'WWER': WWER, 'error_imp_cmb': error_imp_cmb}



'''
Prepares a HTML report

input:
stats: all the statistics collected
reference_text: reference text
outfile: path where the HTML report is saved
logger: logger

output:
returns none
'''
def prepare_HTML(stats, reference_text, outfile, logger):
	logger.info("Preparing the HTML report.")
	template = open(HTML_TEMPLATE).read()
	words = reference_text.split( )

	html_importance_labels = []
	for idx, word in enumerate(reference_text.split()):
		imp_score = stats['imp_label'][idx]
		html_template = "<span class='imp%d'>%s</span>" % (imp_score, word)
		html_importance_labels.append(html_template)

	with open(outfile, 'w') as f:

		#Reference File
		template = template.replace('%reference_filename', stats['reference_filename'])

		#Hypothesis File
		template = template.replace('%hypothesis_filename', stats['hypothesis_filename'])

		#Processed on
		template = template.replace('%timestamp', stats['time_id'])

		#Total words spoken
		template = template.replace('%num_reference_words', str(stats['num_ref_words']))

		#Total words recognized
		template = template.replace('%num_hypothesized_words', str(stats['num_hyp_words']))

		#Word Importance Labeling Output
		template = template.replace('%text_with_word_importance_labels', ' '.join(html_importance_labels))

		#Total number of errors
		template = template.replace("%num_errors", str(stats['num_errors']))

		#Number of substitution errors
		template = template.replace("%num_substitution", str(stats['num_substitution']))

		#Number of deletion errors
		template = template.replace("%num_deletion", str(stats['num_deletion']))

		#Number of insertion errors
		template = template.replace("%num_insertion", str(stats['num_insertion']))

		#Word Error Rate (WER)
		template = template.replace("%WER", str(stats['WER']))

		#Weighted Word Error Rate (WWER)
		template = template.replace("%WWER", str(stats['WWER']))

		#Graph
		template = template.replace("%word_importance_labels", str(['Score: %d' % i for i in range(6)]))
		template = template.replace("%error_count_labels", str(stats['error_imp_cmb'].values()))
		template = template.replace("%count_labels", str(stats['counts'].values()))

		f.write(template)

	logger.info("HTML reported generated and saved at: %s" % outfile)

'''
Computes WER

input:
model: word importance model
reference_file: dict containing file information
and text for the reference transcript
hypothesis_file: dict containing file information
and text for reference transcript
process_word: function to process the word to
adhere to the model format
logger: logger

output:
none
'''
def process_files(model, reference_file, hypothesis_file, process_word, logger):
	timestamp = datetime.datetime.now().isoformat()
	reference_filename = reference_file['filename']
	reference_text = reference_file['text']
	filename = reference_filename.split('/')[-1].split( )[0]

	hypothesis_filename = hypothesis_file['filename']
	hypothesis_text = hypothesis_file['text']

	stats = {'filename': filename, 'time_id': timestamp, 'reference_filename': reference_filename,
	'hypothesis_filename': hypothesis_filename}
	
	other_stats = compute_wer(' '.join(reference_text.split( )), hypothesis_text, logger)
	stats = dict(stats, **other_stats)

	other_stats = importance_labeling(model, reference_text, process_word, 'outputs/%s_%s.csv' % (filename, timestamp), logger)
	stats = dict(stats, **other_stats)

	other_stats = compute_wwer(stats['alignments'], stats['imp_label'], stats['num_ref_words'], logger)
	stats = dict(stats, **other_stats)

	prepare_HTML(stats, reference_text, 'outputs/%s_%s.html' % (filename, timestamp), logger)

