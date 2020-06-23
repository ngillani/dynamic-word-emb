import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from collections import defaultdict
import pandas as pd
import numpy as np
import os

from scipy.spatial.distance import cosine

def count_words_per_attribute(
		# input_file='data/radio_data_by_day_mid_aug_mid_sept.csv',
		input_file='data/all_data_1910_through_1990.csv'
	):
	print 'Loading and preparing data ...'
	df = pd.read_csv(input_file)
	data_per_attr = defaultdict(list)
	for i in range(0, len(df)):
		print i

		# if i == 10000: break

		try:
			data_per_attr[df['id'][i]].extend(simple_preprocess(df['text'][i]))
		except Exception as e:
			print df['text'][i], e
			continue

	num_words = [len(data_per_attr[a]) for a in data_per_attr]
	print np.median(num_words)


def train_word2vec_model_radio(
		input_file='data/radio_data_by_city_and_state.csv',
		model_output_file='non_dynamic_models/models_by_city/w2v_city_%s.model'
	):

	print 'Loading and preparing data ...'
	df = pd.read_csv(input_file)
	data_per_day = defaultdict(list)
	for i in range(0, len(df)):
		print i

		# if i == 10000: break

		try:
			data_per_day[df['id'][i]].append(simple_preprocess(df['text'][i]))
		except Exception as e:
			print df['text'][i], e
			continue

	for d in data_per_day:
		print 'Building model for id %s num snippets %s ...' % (d, len(data_per_day[d]))
		model = Word2Vec(
			size=100,
			window=4,
			min_count=1,
			alpha=0.001,
			workers=4,
			negative=3,
			cbow_mean=1,
			sg=0,
			hs=0
		)	

		print 'Training model ...'
		model.build_vocab(data_per_day[d])
		model.train(data_per_day[d], total_examples=len(data_per_day[d]), epochs=3)

		print 'Saving model ...'
		model.delete_temporary_training_data()
		model.save(model_output_file % d)


def align_all_models(
		models_dir='non_dynamic_models/models_by_city/',
		base_model_file='w2v_city_16.model',
		aligned_models_dir='non_dynamic_models/aligned_models_by_city/'
	):
	
	print 'Loading base model ...'
	base_model = Word2Vec.load(models_dir + base_model_file)
	for f in os.listdir(models_dir):
		if f == base_model_file: continue

		print 'Aligning model %s' % f
		other_model = Word2Vec.load(models_dir + f)
		aligned_other_model = smart_procrustes_align_gensim(base_model, other_model)
		aligned_other_model.save(aligned_models_dir + f)


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
	"""Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
	Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
		(With help from William. Thank you!)

	First, intersect the vocabularies (see `intersection_align_gensim` documentation).
	Then do the alignment on the other_embed model.
	Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
	Return other_embed.

	If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
	"""
	
	# patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
	base_embed.init_sims()
	other_embed.init_sims()

	# make sure vocabulary and indices are aligned
	in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

	# get the embedding matrices
	base_vecs = in_base_embed.syn0norm
	other_vecs = in_other_embed.syn0norm

	# just a matrix dot product with numpy
	m = other_vecs.T.dot(base_vecs) 
	# SVD method from numpy
	u, _, v = np.linalg.svd(m)
	# another matrix operation
	ortho = u.dot(v) 
	# Replace original array with modified one
	# i.e. multiplying the embedding matrix (syn0norm)by "ortho"
	other_embed.syn0norm = other_embed.syn0 = (other_embed.syn0norm).dot(ortho)
	return other_embed
	
def intersection_align_gensim(m1,m2, words=None):
	"""
	Intersect two gensim word2vec models, m1 and m2.
	Only the shared vocabulary between them is kept.
	If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
	Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
	These indices correspond to the new syn0 and syn0norm objects in both gensim models:
		-- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
		-- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
	The .vocab dictionary is also updated for each model, preserving the count but updating the index.
	"""

	# Get the vocab for each model
	vocab_m1 = set(m1.wv.vocab.keys())
	vocab_m2 = set(m2.wv.vocab.keys())

	# Find the common vocabulary
	common_vocab = vocab_m1&vocab_m2
	if words: common_vocab&=set(words)

	# If no alignment necessary because vocab is identical...
	if not vocab_m1-common_vocab and not vocab_m2-common_vocab:
		print 'No alignment necessary!'
		return (m1,m2)

	# Otherwise sort by frequency (summed for both)
	common_vocab = list(common_vocab)
	common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count,reverse=True)

	# Then for each model...
	for m in [m1,m2]:
		# Replace old syn0norm array with new one (with common vocab)
		indices = [m.wv.vocab[w].index for w in common_vocab]
		old_arr = m.wv.syn0norm
		new_arr = np.array([old_arr[index] for index in indices])
		m.syn0norm = m.syn0 = new_arr

		# Replace old vocab dictionary with new one (with common vocab)
		# and old index2word with new one
		m.index2word = common_vocab
		old_vocab = m.wv.vocab
		new_vocab = {}
		for new_index,word in enumerate(common_vocab):
			old_vocab_obj=old_vocab[word]
			new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
		m.wv.vocab = new_vocab

	return (m1,m2)


def compute_bias(
		models_dir='non_dynamic_models/models_by_day/',
		file_template='w2v_day_%s.model',
		base_model_id=0,
		min_id=0,
		max_id=32
	):

	base_model = Word2Vec.load(models_dir + file_template % base_model_id)

	### Refugee outsider vs. all adjectives
	# neutral_words = ['refugee', 'refugees', 'asylum', 'migrant', 'migrants', 'immigrant', 'immigrants']
	neutral_words = ['immigrant', 'immigrants', 'mediterranean']
	baseline_group = ['headstrong', 'thankless', 'tactful', 'distrustful', 'quarrelsome', 'effeminate', 'fickle', 'talkative', 'dependable', 'resentful', 'sarcastic', 'unassuming', 'changeable', 'resourceful', 'persevering', 'forgiving', 'assertive', 'individualistic', 'vindictive', 'sophisticated', 'deceitful', 'impulsive', 'sociable', 'methodical', 'idealistic', 'thrifty', 'outgoing', 'intolerant', 'autocratic', 'conceited', 'inventive', 'dreamy', 'appreciative', 'forgetful', 'forceful', 'submissive', 'pessimistic', 'versatile', 'adaptable', 'reflective', 'inhibited', 'outspoken', 'quitting', 'unselfish', 'immature', 'painstaking', 'leisurely', 'infantile', 'sly', 'praising', 'cynical', 'irresponsible', 'arrogant', 'obliging', 'unkind', 'wary', 'greedy', 'obnoxious', 'irritable', 'discreet', 'frivolous', 'cowardly', 'rebellious', 'adventurous', 'enterprising', 'unscrupulous', 'poised', 'moody', 'unfriendly', 'optimistic', 'disorderly', 'peaceable', 'considerate', 'humorous', 'worrying', 'preoccupied', 'trusting', 'mischievous', 'robust', 'superstitious', 'noisy', 'tolerant', 'realistic', 'masculine', 'witty', 'informal', 'prejudiced', 'reckless', 'jolly', 'courageous', 'meek', 'stubborn', 'aloof', 'sentimental', 'complaining', 'unaffected', 'cooperative', 'unstable', 'feminine', 'timid', 'retiring', 'relaxed', 'imaginative', 'shrewd', 'conscientious', 'industrious', 'hasty', 'commonplace', 'lazy', 'gloomy', 'thoughtful', 'dignified', 'wholesome', 'affectionate', 'aggressive', 'awkward', 'energetic', 'tough', 'shy', 'queer', 'careless', 'restless', 'cautious', 'polished', 'tense', 'suspicious', 'dissatisfied', 'ingenious', 'fearful', 'daring', 'persistent', 'demanding', 'impatient', 'contented', 'selfish', 'rude', 'spontaneous', 'conventional', 'cheerful', 'enthusiastic', 'modest', 'ambitious', 'alert', 'defensive', 'mature', 'coarse', 'charming', 'clever', 'shallow', 'deliberate', 'stern', 'emotional', 'rigid', 'mild', 'cruel', 'artistic', 'hurried', 'sympathetic', 'dull', 'civilized', 'loyal', 'withdrawn', 'confident', 'indifferent', 'conservative', 'foolish', 'moderate', 'handsome', 'helpful', 'gentle', 'dominant', 'hostile', 'generous', 'reliable', 'sincere', 'precise', 'calm', 'healthy', 'attractive', 'progressive', 'confused', 'rational', 'stable', 'bitter', 'sensitive', 'initiative', 'loud', 'thorough', 'logical', 'intelligent', 'steady', 'formal', 'complicated', 'cool', 'curious', 'reserved', 'silent', 'honest', 'quick', 'friendly', 'efficient', 'pleasant', 'severe', 'peculiar', 'quiet', 'weak', 'anxious', 'nervous', 'warm', 'slow', 'dependent', 'wise', 'organized', 'affected', 'reasonable', 'capable', 'active', 'independent', 'patient', 'practical', 'serious', 'understanding', 'cold', 'responsible', 'simple', 'original', 'strong', 'determined', 'natural', 'kind']
	target_group = ['illegal', 'illegals', 'devious', 'bizarre', 'venomous', 'erratic', 'barbaric', 'frightening', 'deceitful', 'forceful', 'deceptive', 'envious', 'greedy', 'hateful', 'contemptible', 'brutal', 'monstrous', 'calculating', 'cruel', 'intolerant', 'aggressive', 'monstrous', 'violent']

	all_bias_scores = []

	# For each "document" (i.e., year), compute occupation gender bias 
	for i in range(min_id, max_id):

		if i == base_model_id:
			model = base_model

		else:
			print 'Loading model ...'
			init_model = Word2Vec.load(models_dir + file_template % i)

			print 'Aligning model %s' % i
			model = smart_procrustes_align_gensim(base_model, init_model)
			base_model = model

		print 'Computing bias scores ...'
		bias = _compute_relative_bias_score_categorical(model, neutral_words, baseline_group, target_group)
		all_bias_scores.append(bias)

	print all_bias_scores


def _compute_relative_bias_score_categorical(model, neutral_words, baseline_group, target_group):
	
	# Compute the average word distances for the baseline and refugee groups

	print 'Computing average vector for baseline words ...'
	mean_baseline_vec = _compute_average_vector_categorical(model, baseline_group)

	print 'Computing average vector for target words ...'
	mean_target_vec = _compute_average_vector_categorical(model, target_group)

	# Now, for each neutral world, compute the difference in the relative norms wrt that word and average baseline and target vectors
	print 'Computing relative norm diff ...'
	bias = 0
	for w in neutral_words:
		try:
			neutral_wordvec = model.wv[w]

		except Exception as e: 
			print e
			continue

		# bias += (np.linalg.norm(neutral_wordvec - mean_baseline_vec) - np.linalg.norm(neutral_wordvec - mean_target_vec))
		bias += cosine(neutral_wordvec, mean_baseline_vec) - cosine(neutral_wordvec, mean_target_vec)

	return bias


def _compute_average_vector_categorical(model, group):
	
	group_vecs = []
	for w in group:

		try:
			group_vecs.append(model.wv[w])
		except Exception as e:
			print e
			continue

	# return np.mean(group_vecs)
	return np.mean(group_vecs, axis=0)

if __name__ == "__main__":
	# count_words_per_attribute()
	# train_word2vec_model_radio()
	# align_all_models()
	compute_bias()

