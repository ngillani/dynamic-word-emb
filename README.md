## Dynamic embedding model, forked from the PyTorch implementation of paragraph vectors [here](https://github.com/inejc/paragraph-vectors).


This repository contains the code for the **corrected** version of:  

> Gillani & Levy. 2019. Simple dynamic word embeddings for mapping perceptions in the public sphere. Proceedings of the Third Workshop on Natural Language Processing and Computational Social Science. Pages 94–99.  

Please refer to the corrected version of this paper, not the originally-published version, as the results are qualitatively different in the corrected version.


Quick sketch of important files (most of the code is taken from the paragraph vectors implementation linked above, with modifications where necessary):
- `paragraphvec/models.py`: contains code for categorical dynamic model (class name: DM) and continuous dynamic model (class name: DMSpline — this is a work in progress)
- `paragraphvec/dataset.py`: contains functions for a) preparing various datasets (namely COHA and talk radio), and b) creating and serving batches for training
- `paragraphvec/train.py`: contains code for training model
- `paragraphvec/compute_bias.py`: functions for retrieving vectors and computing bias using the relative norm distance metric from [this paper](https://pnas.org/content/early/2018/03/30/1720347115).
- `paragraphvec/non_dynamic_embeddings.py`: functions for producing the attribute-specific w2v models, used as a benchmark in figure 3(a)

### Instructions for replicating the main results from the corrected version of [Simple dynamic word embeddings for mapping perceptions in the public sphere](https://arxiv.org/pdf/1904.03352.pdf)

- Figure 2
	- [COHA data](https://www.dropbox.com/s/k92xm2ykjhof8a6/all_data_1910_through_1990.csv?dl=0).
	- Command to train model: 

	`CUDA_VISIBLE_DEVICES=0 python train.py start --data_file_name 'all_data_1910_through_1990.csv' --num_epochs 3 --batch_size 128 --num_noise_words 3 --vec_dim 100 --context_size 4 --lr 1e-3 --model_ver 'dm'`

	- Explanation of parameters:
		- data_file_name => data to be used to train model.  Rows correspond to text entries (e.g. an element in COHA).  The first column contains the attribute ID (e.g. decade of COHA document); the second column contains the text
		- num_epochs => number of epochs to train for.  For COHA, training takes several days per epoch.
		- batch_size => number of data points per batch
		- num_noise_words => number of noise words used in the negative sampling loss function (does not include the ground truth target word)
		- vec_dim => size of each word embedding
		- context_size => number of words to the left and right used per training sample (so, 2 x context_size - both the left and right context - are used in our CBOW training mdoel)
		- lr => learning rate
		- model_ver => model version (in our case, always 'dm' for 'distributed memory', from the fact that we adapt our model from the distributed memory model of paragraph vectors)
	- To produce bias scores, we first ensure that the correct word lists are uncommented in the paragraphvec/compute_bias.py file's "start" function.  Next, we run the following command, replacing the model file name with the correct one outputted by the command above: 

	`python compute_bias.py start --data_file_name 'all_data_1910_through_1990.csv' --model_file_name <TRAINED_MODEL_FILE_NAME>`

	- The above step should print out the resultant bias scores per decade (just make sure `start` is invoking the right function in the file).  These scores are computed for gender and ethnic occupation and compared to the scores produced in Garg et al. (which were provided to us directly from the authors).

- Figure 3
    - To produce a) and c), we run `find_nearest_words_per_attr_value` found in `compute_bias.py`
    - To produce b) and d), we run `find_nearest_words_non_dynamic` found in `compute_bias.py`

- Figure 4
	- Talk radio [timeseries data](https://www.dropbox.com/s/2c678dhb1w2q136/radio_data_by_day_mid_aug_mid_sept.csv?dl=0).
	- Command to train model: 

	`CUDA_VISIBLE_DEVICES=0 python train.py start --data_file_name 'radio_data_by_day_mid_aug_mid_sept.csv' --num_epochs 3 --batch_size 128 --num_noise_words 3 --vec_dim 100 --context_size 4 --lr 1e-3 --model_ver 'dm'`

	- Explanation of parameters: same as above for Figure 2
	- To produce bias scores, we first ensure that the correct word lists (namely, the refugee-related ones) are uncommented in the paragraphvec/compute_bias.py file's "start" function.  Next, we run the following command, replacing the model file name with the correct one outputted by the command above: 

	`python compute_bias.py start --model_file_name <TRAINED_MODEL_FILE_NAME>  `

	- Bias scores will the printed to the terminal.  We divide these by |R| (the number of refugee-related terms, which in our case is 7) to produce the values for Figure 4(b).
    - To produce the values for Figure 4(a), we run

	`PYTHONPATH=. python non_dynamic_embeddings.py`

	- The above command will first train a word2vec model per day of talk radio data.  Next, it will compute bias scores by aligning each of the day-by-day w2v models using Procrustes alignment.
	- Bias scores will the printed to the terminal.  Again, we divide these by |R|.



