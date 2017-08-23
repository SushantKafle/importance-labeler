import os
from annotation import AnnotationDataset
from utils import create_folders, load_vocab, \
get_word_vectors, get_feat_vectors, process_word, process_tag
from wimp import WImpModel
from config import config

create_folders(config.model_out)

vocab_words = load_vocab(config.words_vocab_path)
feats = get_feat_vectors(config.feats_file)

word_processor = process_word(vocab_words)
tag_processor = process_tag()

dev = AnnotationDataset(config.dev_data, vocab_words, word_processor, tag_processor)
test = AnnotationDataset(config.test_data, vocab_words, word_processor, tag_processor)
train = AnnotationDataset(config.train_data, vocab_words, word_processor, tag_processor)

model = WImpModel(config, vocab_words, feats)
model.setup()

model.train(train, dev)
model.evaluate(test)
model.interactive_shell(word_processor)


