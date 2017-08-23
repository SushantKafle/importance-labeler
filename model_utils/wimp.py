import tensorflow as tf 
import numpy as np
import csv
from utils import get_logger, create_folders, minibatches, pad_sequences
from feats import get_feats, lookup_feats, get_feature

class WImpModel(object):

	def __init__(self, config, words, feats, logger = None):

		#reset the graph
		tf.reset_default_graph()
		tf.set_random_seed(config.random_seed)

		self.config = config
		self.feats = feats
		self.words = words
		self.logger = get_logger(self.config.log_out) if not logger else logger
		self.id_to_word = self.idx_to_word(self.words)
		self.ntags = self.config.ntags
		
		self.record_config()


	def idx_to_word(self, vocab):
		id_to_word = {idx: word for word, idx in vocab.iteritems()}
		return id_to_word

	def record_config(self):
		self.logger.info("feats_dim: %d" % self.config.hand_feats_dim)
		self.logger.info("num_epoch: %d" % self.config.num_epoch)
		self.logger.info("initial_lr: %d" % self.config.lr)
		self.logger.info("dropout: %f" % self.config.dropout)

	def add_summary(self, sess):
		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.config.model_out, sess.graph)

	def setup(self):
		self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
		self.word_feats = tf.placeholder(tf.float32, shape=[None, None, None], name="word_feats")
		self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
		self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

		self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
		self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

		hand_feats = tf.nn.dropout(self.word_feats, self.dropout)
		ntime = tf.shape(hand_feats)[1]

		hand_feats = tf.reshape(hand_feats, [-1, self.config.hand_feats_dim])
		with tf.variable_scope("projection-layer"):
			W = tf.get_variable("W", shape=[self.config.hand_feats_dim, self.ntags], dtype=tf.float32)
			b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32)
			proj = tf.matmul(hand_feats, W) + b

		self.logits = tf.reshape(proj, [-1, ntime, self.ntags])

		log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
	            self.logits, self.labels, self.sequence_lengths)
		self.loss = tf.reduce_mean(-log_likelihood)

		with tf.variable_scope("trainer"):
			optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, rho=0.9)
			self.train_ = optimizer.minimize(self.loss)

	
	def get_feed(self, word_ids, batch_id, labels = None, lr = None, dropout = None, training=False, lookup=True):
		word_ids, sequence_lengths = pad_sequences(word_ids, -1)
		nwords = len(word_ids[0])

		train_hinge = 0 if self.config.train_hinge < 0 else self.config.train_hinge
		batch_id = train_hinge + batch_id if not training else batch_id

		if lookup and train_hinge > 0:
			word_feats = lookup_feats(self.feats, word_ids, 
			nwords, batch_id)
		else:
			word_feats = get_feats(word_ids, self.id_to_word)

		feed = {
			self.word_ids : word_ids,
			self.sequence_lengths : sequence_lengths,
			self.word_feats : word_feats
		}

		if labels is not None:
			labels, _ = pad_sequences(labels, 0)
			feed[self.labels] = labels

		if lr is not None:
			feed[self.lr] = lr

		if dropout is not None:
			feed[self.dropout] = dropout

		return feed, sequence_lengths


	def predict_batch(self, sess, word_ids, batch_id, is_training = False, is_dev = False):
		dropout = config.dropout if is_training else 1.0
		feed, sequence_lengths = self.get_feed(word_ids = word_ids, batch_id = batch_id, 
		dropout = dropout, training = is_training, lookup = (is_training or is_dev))

		viterbi_sequences = []
		logits, transition_params = sess.run([self.logits, self.transition_params], feed_dict = feed)
		for logit, sequence_length in zip(logits, sequence_lengths):
			logit = logit[:sequence_length]
			viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
				logit, transition_params)
			viterbi_sequences += [viterbi_sequence]

		return viterbi_sequences, sequence_lengths
		
	def performance_eval(self, sess, test, is_dev = True):
		accs, all_labels, all_labels_pred = [], [], []
		main_predicted_count, main_total_count, main_correct_count = 0., 0., 0.
		for i, (word_ids, labels) in enumerate(minibatches(test, self.config.batch_size)):
			labels_pred, sequence_lengths = self.predict_batch(sess, word_ids,
				i * self.config.batch_size, is_dev = is_dev, is_training = False)

			all_labels_pred.append(labels_pred)
			all_labels.append(labels)

			for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
				lab = lab[:length]
				lab_pred = lab_pred[:length]
				accs += map(lambda (a, b): a == b, zip(lab, lab_pred))
				main_predicted_count += sum(map(lambda (a): a == self.config.main_label, lab_pred))
				main_total_count += sum(map(lambda (a): a == self.config.main_label, lab))
				main_correct_count += sum(map(lambda (a, b): (a == self.config.main_label) * (b == self.config.main_label), zip(lab_pred, lab)))

		acc = np.mean(accs)
		p = (float(main_correct_count)/ float(main_predicted_count)) if (main_predicted_count > 0) else 0.0
		r = (float(main_correct_count)/ float(main_total_count)) if (main_total_count > 0) else 0.0
		f = (2.0 * p * r / (p + r)) if (p+r > 0.0) else 0.0
		f05 = ((1 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (p+r > 0.0) else 0.0

		return acc, f05


	def run(self, sess, train, dev, epoch):
		nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
		for i, (word_ids, labels) in enumerate(minibatches(train, self.config.batch_size)):
			feed, sequence_lengths = self.get_feed(word_ids = word_ids, batch_id = i * self.config.batch_size,
			 labels = labels, lr = self.config.lr, dropout = self.config.dropout, training = True)
			#print (np.asarray(feed[self.word_feats]))
			_, train_loss = sess.run([self.train_, self.loss], feed_dict=feed)
			self.logger.info("Train loss: %f" % train_loss)

		acc, f05 = self.performance_eval(sess, dev, is_dev = True)
		self.logger.info("dev accuracy: %f, f05: %f" % (acc, f05))
		return acc, f05


	def train(self, train, dev):
		saver = tf.train.Saver()
		best_accuracy = 0
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.add_summary(sess)
			for epoch in range(self.config.num_epoch):
				self.logger.info("Epoch %d out of %d" % (epoch + 1, self.config.num_epoch))
				accuracy, f05 = self.run(sess, train, dev, epoch)
				#self.config.lr *= self.config.lr_decay

				if accuracy >= best_accuracy:
					best_accuracy = accuracy
					create_folders(self.config.model_out)
					saver.save(sess, self.config.model_out)
		self.logger.info("best accuracy: %f" % (best_accuracy))


	def evaluate(self, test):
		saver = tf.train.Saver()
		with tf.Session() as sess:
			self.logger.info("Evaluation of the test sets..")
			saver.restore(sess, self.config.model_out)
			acc, f05 = self.performance_eval(sess, test, is_dev = True)
			self.logger.info("test accuracy: %f" % acc)

	def interactive_shell(self, process_word):
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, self.config.model_out)
			self.logger.info("This is an interactive session, enter a sentence:")
			while True:
				try:
					sentence = raw_input("\ninput> ")
					words_raw = sentence.strip().split(" ")
					words = map(lambda x: process_word(x), words_raw)
					word_ids = [self.words[word] for word in words]
					preds, _ = self.predict_batch(sess, [word_ids], -1)
					print (preds)
				except EOFError:
					print ("Closing session.")
					break


	def label_text(self, text, process_word):
		saver = tf.train.Saver()
		output = []
		with tf.Session() as sess:
			saver.restore(sess, self.config.model_out)
			for sentence in text:
				words_raw = sentence.strip().split(" ")
				words = map(lambda x: process_word(x), words_raw)
				word_ids = [self.words[word] for word in words]
				preds, _ = self.predict_batch(sess, [word_ids], -1)
				
				for word, prediction in zip(words, preds[0]):
					output.append(int(prediction))

		return output

				
