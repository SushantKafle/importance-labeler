import tensorflow as tf 
import numpy as np
from utils import get_logger, create_folders, minibatches, pad_sequences
from feats import get_feats

class WImpModel(object):

	def __init__(self, config, embeddings, words, tags):

		#reset the graph
		tf.reset_default_graph()
		tf.set_random_seed(config.random_seed)

		self.config = config
		self.embeddings = embeddings
		self.words = words
		self.tags = tags
		self.ntags = len(self.tags)
		self.logger = get_logger(self.config.log_out)
		self.record_config()
		self.idx_to_word = self._idx_to_word(self.words)
		self.idx_to_tag = self._idx_to_word(self.tags)


	def _idx_to_word(self, vocab):
		def f(vocab):
			id_to_word = {idx: word for word in vocab.iteritems()}
			return id_to_word

		return f

	def record_config(self):
		self.logger.info("word_embedding_dim: %d" % self.config.word_embedding_dim)
		self.logger.info("word_feats_dim: %d" % self.config.word_feats_dim)
		self.logger.info("hidden_layer_size: %d" % self.config.hidden_layer_size)
		self.logger.info("narrow_layer_size: %d" % self.config.narrow_layer_size)
		self.logger.info("num_epoch: %d" % self.config.num_epoch)

	def add_summary(self, sess):
		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.config.model_out, sess.graph)

	def setup(self):

		self.word_ids = tf.placeholder(tf.int32, shape=[None, None], name="word_ids")
		self.word_feats = tf.placeholder(tf.float32, shape=[None, None, None], name="word_feats")
		self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")
		self.word_lengths = tf.placeholder(tf.int32, shape=[None, None], name="word_lengths")
		self.labels = tf.placeholder(tf.int32, shape=[None, None], name="labels")

		self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
		self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

		embeddings = tf.Variable(self.embeddings, name="embeddings", dtype=tf.float32, trainable=False)
		word_embeddings = tf.nn.embedding_lookup(embeddings, self.word_ids, name="word_embeddings")
		self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)
		ntime = tf.shape(self.word_embeddings)[1]

		word_feats = tf.reshape(self.word_feats, [-1, self.config.word_feats_dim])
		with tf.variable_scope("feat-transform"):
			W = tf.get_variable("W", shape=[self.config.word_feats_dim, 
			self.config.feats_hidden_layer], dtype=tf.float32)
			b = tf.get_variable("b", shape=[self.config.feats_hidden_layer], dtype=tf.float32)
			word_feats_transform = tf.tanh(tf.matmul(word_feats, W) + b)
			word_feats_transform = tf.reshape(word_feats_transform, [-1, ntime, self.config.feats_hidden_layer])

		_input = tf.concat([word_feats_transform, self.word_embeddings], axis=-1)
		_input = tf.reshape(_input, [-1, self.config.feats_hidden_layer + self.config.word_embedding_dim])
		with tf.variable_scope("nonlinear-layer"):
			W = tf.get_variable("W", shape=[self.config.feats_hidden_layer + self.config.word_embedding_dim, 
			self.config.hidden_layer_size], dtype=tf.float32)
			b = tf.get_variable("b", shape=[self.config.hidden_layer_size], dtype=tf.float32)
			hidden_layer_output = tf.tanh(tf.matmul(_input, W) + b)

		'''with tf.variable_scope("narrow-layer"):
				W = tf.get_variable("W", shape=[self.config.feats_hidden_layer + self.config.word_embedding_dim, self.config.narrow_layer_size], dtype=tf.float32)
				b = tf.get_variable("b", shape=[self.config.narrow_layer_size], dtype=tf.float32)
				narrow_layer_output = tf.tanh(tf.matmul(_input, W) + b)'''

		with tf.variable_scope("projection-layer"):
			W = tf.get_variable("W", shape=[self.config.hidden_layer_size, self.ntags], dtype=tf.float32)
			b = tf.get_variable("b", shape=[self.ntags], dtype=tf.float32)
			proj = tf.matmul(hidden_layer_output, W) + b

		self.logits = tf.reshape(proj, [-1, ntime, self.ntags])

		if not self.config.crf:
			self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1), tf.int32)
			losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
			mask = tf.sequence_mask(self.sequence_lengths)
			self.loss = tf.reduce_mean(tf.boolean_mask(losses, mask))
		else:
			log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
	            self.logits, self.labels, self.sequence_lengths)
			self.loss = tf.reduce_mean(-log_likelihood)

		#Setting up trainer
		with tf.variable_scope("trainer"):
			#optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr, rho=0.9)
			optimizer = tf.train.AdamOptimizer(self.lr)
			self.train_ = optimizer.minimize(self.loss)

	def get_feed(self, words, words_, labels = None, lr = None, dropout = None):
		words_, _ = pad_sequences(words_, 0)
		word_feats = get_feats(words_)
		word_ids, sequence_lengths = pad_sequences(words, 0)
		

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


	def predict_batch(self, sess, words, words_):
		feed, sequence_lengths = self.get_feed(words=words, words_=words_, dropout=1.0)

		if self.config.crf:
			viterbi_sequences = []
			logits, transition_params = sess.run([self.logits, self.transition_params], 
				feed_dict = feed)

			for logit, sequence_length in zip(logits, sequence_lengths):
				logit = logit[:sequence_length]
				viterbi_sequence, viterbi_score = tf.contrib.crf.viterbi_decode(
					logit, transition_params)
				viterbi_sequences += [viterbi_sequence]

			return viterbi_sequences, sequence_lengths
		else:
			labels_pred = sess.run(self.labels_pred, feed_dict=feed)
			return labels_pred, sequence_lengths

	def performance_eval(self, sess, test, tags):
		accs = []
		main_predicted_count, main_total_count, main_correct_count = 0., 0., 0.
		for words_, words, labels in minibatches(test, self.config.batch_size):
			labels_pred, sequence_lengths = self.predict_batch(sess, words, words_)
			for lab, lab_pred, length in zip(labels, labels_pred, sequence_lengths):
				lab = lab[:length]
				lab_pred = lab_pred[:length]
				accs += map(lambda (a, b): a == b, zip(lab, lab_pred))
				main_predicted_count += sum(map(lambda (a): a == tags[self.config.main_label], lab_pred))
				main_total_count += sum(map(lambda (a): a == tags[self.config.main_label], lab))
				main_correct_count += sum(map(lambda (a, b): (a == tags[self.config.main_label]) * (b == tags[self.config.main_label]), zip(lab_pred, lab)))
		acc = np.mean(accs)
		p = (float(main_correct_count)/ float(main_predicted_count)) if (main_predicted_count > 0) else 0.0
		r = (float(main_correct_count)/ float(main_total_count)) if (main_total_count > 0) else 0.0
		f = (2.0 * p * r / (p + r)) if (p+r > 0.0) else 0.0
		f05 = ((1 + 0.5 * 0.5) * p * r / ((0.5 * 0.5 * p) + r)) if (p+r > 0.0) else 0.0
		return acc, f05


	def run(self, sess, train, dev, tags, epoch):
		nbatches = (len(train) + self.config.batch_size - 1) / self.config.batch_size
		for i, (words_, words, labels) in enumerate(minibatches(train, self.config.batch_size)):
			feed, sequence_lengths = self.get_feed(words, words_, labels, self.config.lr, self.config.dropout)
			_, train_loss = sess.run([self.train_, self.loss], feed_dict=feed)
			self.logger.info("Train loss: %f" % train_loss)

		acc, f05 = self.performance_eval(sess, dev, tags)
		self.logger.info("dev accuracy: %f, f05: %f" % (acc, f05))
		return acc, f05
			


	def train(self, train, dev, tags):
		saver = tf.train.Saver()
		best_f05 = 0
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.add_summary(sess)
			for epoch in range(self.config.num_epoch):
				self.logger.info("Epoch %d out of %d" % (epoch + 1, self.config.num_epoch))
				accuracy, f05 = self.run(sess, train, dev, tags, epoch)
				self.config.lr *= self.config.lr_decay

				if f05 >= best_f05:
					best_f05 = f05
					create_folders(self.config.model_out)
					saver.save(sess, self.config.model_out)


	def evaluate(self, test, tags):
		saver = tf.train.Saver()
		with tf.Session() as sess:
			self.logger.info("Evaluating of the test sets..")
			saver.restore(sess, self.config.model_out)
			acc, f05 = self.performance_eval(sess, test, tags)
			self.logger.info("test accuracy: %f, f05: %f" % (acc, f05))












