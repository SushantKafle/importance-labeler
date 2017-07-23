import os
from utils import UNK, NUM

class AnnotationDataset(object):
	
	def __init__(self, path, word_vocab, tag_vocab):
		self.path = path
		self.word_vocab = word_vocab
		self.tag_vocab = tag_vocab
		self.length = None


	def __iter__(self):
		if os.path.isfile(os.path.join(self.path)):
			with open(self.path) as file_obj:
				words, word_ids, tags = [], [], []
				for line in file_obj:
					if line.strip() != '':
						word, tag = line.split('\t')
						word = word.strip()
						tag = tag.strip()
						words += [word]

						word = word.lower()
						if word.isdigit():
							word = NUM
						if word not in self.word_vocab:
							word = UNK

						word_ids += [self.word_vocab[word]]
						tags += [self.tag_vocab[tag]]
					else:
						if len(words) != 0:
							yield words, word_ids, tags
							words, word_ids, tags = [], [], []


	def __len__(self):
		if self.length is None:
			self.length = 0
			for _ in self:
				self.length += 1
		return self.length


	