class config():
	script_path = 'data/scripts'
	train_data = 'data/train.txt'
	dev_data = 'data/test.txt'
	test_data = 'data/test.txt'
	feats_file = 'data/feats.npz'

	train_hinge = 526

	words_vocab_path = 'data/words.txt'
	annotation_src = 'data/annotation.csv'
	
	model_out = 'results/model'
	log_out = 'results/log.log'

	main_label = 5
	random_seed = 1

	hand_feats_dim = 32
	ntags = 6
	
	num_epoch = 100
	batch_size = 20

	lr = 1
	dropout = 1
	
