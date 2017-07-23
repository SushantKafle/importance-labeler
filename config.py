class config():
	script_src = 'data/script.txt'
	train_data = 'data/train.txt'
	dev_data = 'data/test.txt'
	test_data = 'data/test.txt'

	words_vocab_path = 'data/words.txt'
	tags_vocab_path = 'data/tags.txt'

	annotation_src = 'data/annotation.csv'
	cmp_embeddings_src = 'data/senna-cmp.npz'

	model_out = 'results/model'
	log_out = 'results/log.txt'
	main_label = 'nimp'

	random_seed = 1

	word_embedding_dim = 50
	word_feats_dim = 27
	feats_hidden_layer = 5

	hidden_layer_size = 40
	narrow_layer_size = 10

	num_epoch = 20
	batch_size = 20

	lr = 1
	lr_decay = 0.9
	dropout = 0.5

	crf = True
	


	senna_home = "data/senna/"
	
	
