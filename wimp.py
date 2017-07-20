import tensorflow as tf 
import numpy as np
from utils import get_logger

class WImpModel(object):

	def __init__(self, config, embeddings, words, tags):

		#reset the graph
		tf.reset_default_graph()
		tf.set_random_seed(config.random_seed)

		self.config = config
		self.embeddings = embeddings
		self.words = words
		self.tags = tags
		self.logger = get_logger(self.config.log_file)
		self.record_config()



	def record_config(self):
		self.logger.info("word_embedding_size: %d" % self.config.word_embedding_size)
		#blah blah



