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

from copy import deepcopy

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


def prep_data_function(data):

	data_batches = data[0]
	num_context_words = data[1]
	num_noise_words = data[2]
	word_to_ind_dict = dict(data[3])
	ind_to_word_dict = dict([(word_to_ind_dict[w], w) for w in word_to_ind_dict])
	word_to_freq_dict = dict(data[4])
	probs = np.zeros(len(word_to_freq_dict))

	for word, freq in word_to_freq_dict.items():
		if word in word_to_ind_dict:
			probs[word_to_ind_dict[word]] = freq

	probs = np.power(probs, 0.75)
	probs /= np.sum(probs)

	sample_noise = lambda: np.random.choice(
		probs.shape[0], num_noise_words, p=probs).tolist()

	all_doc_ids = []
	all_context_ids = []
	all_target_noise_ids = []

	for val, doc in enumerate(data_batches):
		print (val)
		# if val == 5: break
		doc_text = []
		for w in doc.text:
			if not w in word_to_ind_dict: continue
			doc_text.append(w)
		curr_len = len(doc_text) 
		if len(doc_text) < 2 * num_context_words + 1: continue
		curr_ind = num_context_words
		while curr_ind < len(doc_text) - num_context_words:
			curr_doc_id = int(doc.id)
			curr_context_ids = [word_to_ind_dict[doc_text[i]] for i in range(curr_ind - num_context_words, curr_ind + num_context_words + 1) if i != curr_ind]
			current_noise = sample_noise()
			current_noise.insert(0, word_to_ind_dict[doc_text[curr_ind]])
			
			all_doc_ids.append(curr_doc_id)
			all_context_ids.append(curr_context_ids)
			all_target_noise_ids.append(current_noise)

			curr_ind += 1

		# if val == 2:
		# 	print (all_context_ids, all_doc_ids, all_target_noise_ids)
		# 	for i in all_context_ids[-1]:
		# 		print (i, ind_to_word_dict[i])
		# 	print doc.id, doc_text
		# 	exit()

	# print ('returning doc ids: ', len(all_doc_ids))
	return all_doc_ids, all_context_ids, all_target_noise_ids


def load_and_cache_data(
	   data_file_root='tiny_all_data_1910_through_1990.csv',
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
			doc_ids, context_ids, target_noise_ids, word_to_ind_dict = pickle.load(f)
		print('data loaded from cache!')
	
	else:
		print('Preparing data ...')
		
		N_THREADS = 5

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
		# sample_noise = init_noise_distribution(curr_vocab, num_noise_words)

		print ('curr vocab size: ', len(curr_vocab.stoi))

		batch_size = int(len(dataset) / N_THREADS)

		arg_iterables = []
		for i in range(0, N_THREADS + 1):
			data_batch = dataset[batch_size*i:batch_size*(i+1)]
			vocab_ind_copy = [(w, curr_vocab.stoi[w]) for w in curr_vocab.stoi]
			vocab_freq_copy = [(w, curr_vocab.freqs[w]) for w in curr_vocab.stoi]
			arg_iterables.append((data_batch, num_context_words, num_noise_words, vocab_ind_copy, vocab_freq_copy))

		from multiprocessing import Pool

		# Make arrays of batch entries
		doc_ids = []
		context_ids = []
		target_noise_ids = []

		p = Pool(N_THREADS)

		for curr_doc_ids, curr_context_ids, curr_target_noise_ids in p.map(prep_data_function, arg_iterables):
			doc_ids.extend(curr_doc_ids)
			context_ids.extend(curr_context_ids)
			target_noise_ids.extend(curr_target_noise_ids)

		p.close()
		p.join()

		# print (doc_ids, len(doc_ids))
		# exit()

		print ('Writing to disk ...')
		word_to_ind_dict = dict(curr_vocab.stoi)
		print (word_to_ind_dict)
		with open(join(DATA_DIR, prepared_data_file), 'wb') as f:
			pickle.dump((doc_ids, context_ids, target_noise_ids, word_to_ind_dict), f)
			print('Data written to disk!')

	# tensorize everything
	data_to_return = {'doc_ids': doc_ids, 'context_ids': context_ids, 'target_noise_ids': target_noise_ids}
	for d in data_to_return:
		data_to_return[d] = torch.LongTensor(data_to_return[d])

	return data_to_return['doc_ids'], data_to_return['context_ids'], data_to_return['target_noise_ids'], word_to_ind_dict
	

if __name__ == "__main__":
	load_and_cache_data()
