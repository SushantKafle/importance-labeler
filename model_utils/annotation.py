import os


class AnnotationDataset(object):
	
	def __init__(self, path, word_vocab, process_word, process_tag):
		self.path = path
		self.word_vocab = word_vocab
		self.length = None
		self.process_word = process_word
		self.process_tag = process_tag


	def __iter__(self):
		if os.path.isfile(os.path.join(self.path)):
			with open(self.path) as file_obj:
				word_ids, tags = [], []
				for line in file_obj:
					if line.strip() != '':
						word, tag = line.split('\t')
						word = self.process_word(word)
						tag = self.process_tag(tag)
						word_ids += [self.word_vocab[word]]
						tags += [tag]
					else:
						if len(word_ids) != 0:
							yield word_ids, tags
							word_ids, tags = [], []


	def __len__(self):
		if self.length is None:
			self.length = 0
			for _ in self:
				self.length += 1
		return self.length


	