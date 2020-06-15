import os
from os.path import join
import re
import pickle
import sys
import csv

import torch
from torchtext.vocab import Vocab
from torchtext.data import Field, TabularDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import pandas as pd
import pdb
import numpy as np

from utils import DATA_DIR, read_dict

def make_dataloader(data, batch_size, shuffle=True, sampler=None):
	data = TensorDataset(*data)
	data_loader = DataLoader(data, shuffle=shuffle, sampler=sampler, batch_size=batch_size)
	return data_loader


def _tokenize_str(str_):
    # keep only alphanumeric and punctations
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    # lower case
    return str_.strip().lower().split()


def init_noise_distribution(vocab, num_noise_words):
	# we use a unigram distribution raised to the 3/4rd power,
	# as proposed by T. Mikolov et al. in Distributed Representations
	# of Words and Phrases and their Compositionality
	probs = np.zeros(len(vocab))

	for word, freq in vocab.freqs.items():
		if word in vocab.stoi:
			probs[vocab.stoi[word]] = freq

	probs = np.power(probs, 0.75)
	probs /= np.sum(probs)

	sample_noise = lambda: np.random.choice(
		probs.shape[0], num_noise_words, p=probs).tolist()
	
	return sample_noise


def load_and_cache_data(
	   data_file_root='all_data_1910_through_1990.csv',
	   num_context_words=4,
	   num_noise_words=3,
	   min_word_freq=10
	):

	data_file_root = data_file_root.split('.')[0]
	raw_data_file = data_file_root + '.csv'
	prepared_data_file = data_file_root + '.p'

	print('Loading data {} ...'.format(join(DATA_DIR, raw_data_file)))
	
	if os.path.isfile(join(DATA_DIR, prepared_data_file)):
	# if False:
		print('Loading data from catch ...')
		with open(join(DATA_DIR, prepared_data_file), 'rb') as f:
			doc_ids, context_ids, target_noise_ids, word_to_ind_dict  = pickle.load(f)
		print('data loaded from cache!')
	
	else:
		print('Preparing data ...')
		
		csv.field_size_limit(sys.maxsize)

		file_path = join(DATA_DIR, raw_data_file)
		id_field = Field(sequential=False, use_vocab=False, dtype=torch.int)
		text_field = Field(pad_token=None, tokenize=_tokenize_str)

		dataset = TabularDataset(
			path=file_path,
			format='csv',
			fields=[('id', id_field), ('text', text_field)],
			skip_header=True)

		text_field.build_vocab(dataset, min_freq=min_word_freq)
		curr_vocab = dataset.fields['text'].vocab
		sample_noise = init_noise_distribution(curr_vocab, num_noise_words)

		# Make arrays of batch entries
		doc_ids = []
		context_ids = []
		target_noise_ids = []

		print ('curr vocab size: ', len(curr_vocab.stoi))
		
		for val, doc in enumerate(dataset):
			print (val)
			doc_text = []
			for w in doc.text:
				if not w in curr_vocab.stoi: continue
				doc_text.append(w)
			curr_len = len(doc_text) 
			if len(doc_text) < 2 * num_context_words + 1: continue
			curr_ind = num_context_words
			while curr_ind < len(doc_text) - num_context_words:
				curr_doc_id = int(doc.id)
				curr_context_ids = [curr_vocab.stoi[doc_text[i]] for i in range(curr_ind - num_context_words, curr_ind + num_context_words + 1) if i != curr_ind]
				current_noise = sample_noise()
				current_noise.insert(0, curr_vocab.stoi[doc_text[curr_ind]])
				
				doc_ids.append(curr_doc_id)
				context_ids.append(curr_context_ids)
				target_noise_ids.append(current_noise)

				curr_ind += 1
			
			# if val == 5:
			# 	print (context_ids, doc_ids, target_noise_ids)
			# 	for i in context_ids[-1]:
			# 		print (i, curr_vocab.itos[i])
			# 	print doc.id, doc_text
			# 	exit()


		word_to_ind_dict = curr_vocab.stoi
		with open(join(DATA_DIR, prepared_data_file), 'wb') as f:
			pickle.dump((doc_ids, context_ids, target_noise_ids, word_to_ind_dict), f)
			print('Data written to disk')

	# tensorize everything
	data_to_return = {'doc_ids': doc_ids, 'context_ids': context_ids, 'target_noise_ids': target_noise_ids}
	for d in data_to_return:
		data_to_return[d] = torch.LongTensor(data_to_return[d])

	return data_to_return['doc_ids'], data_to_return['context_ids'], data_to_return['target_noise_ids'], word_to_ind_dict

if __name__ == "__main__":
	load_and_cache_data()
