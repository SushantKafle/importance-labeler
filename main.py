import os
from annotation import AnnotationDataset
from utils import create_folders, load_vocab, get_word_vectors
from wimp import WImpModel
from config import config

create_folders(config.model_out)

vocab_words = load_vocab(config.words_vocab_path)
vocab_tags = load_vocab(config.tags_vocab_path)

embeddings = get_word_vectors(config.cmp_embeddings_src)

dev = AnnotationDataset(config.dev_data, vocab_words, vocab_tags)
test = AnnotationDataset(config.test_data, vocab_words, vocab_tags)
train = AnnotationDataset(config.train_data, vocab_words, vocab_tags)

model = WImpModel(config, embeddings, vocab_words, vocab_tags)
model.setup()

model.train(train, dev, vocab_tags)
model.evaluate(test, vocab_tags)


