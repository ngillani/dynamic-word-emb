import csv
import re
from os.path import join

import fire
import torch
import numpy as np
import json
from collections import defaultdict

from data import load_dataset
from dataset import load_and_cache_data
from models import DM, DBOW, DMSpline
from utils import DATA_DIR, MODELS_DIR
from scipy.spatial.distance import cosine


def start(model_file_name, model_ver):
    """Saves trained paragraph vectors to a csv file in the *data* directory.

    Parameters
    ----------
    data_file_name: str
        Name of a file in the *data* directory that was used during training.

    model_file_name: str
        Name of a file in the *models* directory (a model trained on
        the *data_file_name* dataset).
    """

    ### Female vs. Male wrt occupation words
    # neutral_words = ['janitor', 'statistician', 'midwife', 'bailiff', 'auctioneer', 'photographer', 'geologist', 'shoemaker', 'athlete', 'cashier', 'dancer', 'housekeeper', 'accountant', 'physicist', 'gardener', 'dentist', 'weaver', 'blacksmith', 'psychologist', 'supervisor', 'mathematician', 'surveyor', 'tailor', 'designer', 'economist', 'mechanic', 'laborer', 'postmaster', 'broker', 'chemist', 'librarian', 'attendant', 'clerical', 'musician', 'porter', 'scientist', 'carpenter', 'sailor', 'instructor', 'sheriff', 'pilot', 'inspector', 'mason', 'baker', 'administrator', 'architect', 'collector', 'operator', 'surgeon', 'driver', 'painter', 'nurse', 'cook', 'engineer', 'retired', 'sales', 'lawyer', 'clergy', 'physician', 'farmer', 'clerk', 'manager', 'guard', 'artist', 'smith', 'official', 'police', 'doctor', 'professor', 'student', 'judge', 'teacher', 'author', 'secretary', 'soldier']
    # # neutral_words = ['statistician', 'auctioneer', 'photographer', 'geologist', 'accountant', 'physicist', 'dentist', 'psychologist', 'supervisor', 'mathematician', 'designer', 'economist', 'postmaster', 'broker', 'chemist', 'librarian', 'scientist', 'instructor', 'pilot', 'administrator', 'architect', 'surgeon', 'nurse', 'engineer', 'lawyer', 'physician', 'manager', 'official', 'doctor', 'professor', 'student', 'judge', 'teacher', 'author']
    # baseline_group = ['he', 'son', 'his', 'him', 'father', 'man', 'boy', 'himself', 'male', 'brother', 'sons', 'fathers', 'men', 'boys', 'males', 'brothers', 'uncle', 'uncles', 'nephew', 'nephews']
    # target_group = ['she', 'daughter', 'hers', 'her', 'mother', 'woman', 'girl', 'herself', 'female', 'sister', 'daughters', 'mothers', 'women', 'girls', 'femen', 'sisters', 'aunt', 'aunts', 'niece', 'nieces']

    ### Asian vs. White wrt occupation words
    # neutral_words = ['janitor', 'statistician', 'midwife', 'bailiff', 'auctioneer', 'photographer', 'geologist', 'shoemaker', 'athlete', 'cashier', 'dancer', 'housekeeper', 'accountant', 'physicist', 'gardener', 'dentist', 'weaver', 'blacksmith', 'psychologist', 'supervisor', 'mathematician', 'surveyor', 'tailor', 'designer', 'economist', 'mechanic', 'laborer', 'postmaster', 'broker', 'chemist', 'librarian', 'attendant', 'clerical', 'musician', 'porter', 'scientist', 'carpenter', 'sailor', 'instructor', 'sheriff', 'pilot', 'inspector', 'mason', 'baker', 'administrator', 'architect', 'collector', 'operator', 'surgeon', 'driver', 'painter', 'nurse', 'cook', 'engineer', 'retired', 'sales', 'lawyer', 'clergy', 'physician', 'farmer', 'clerk', 'manager', 'guard', 'artist', 'smith', 'official', 'police', 'doctor', 'professor', 'student', 'judge', 'teacher', 'author', 'secretary', 'soldier']
    # # neutral_words = ['statistician', 'auctioneer', 'photographer', 'geologist', 'accountant', 'physicist', 'dentist', 'psychologist', 'supervisor', 'mathematician', 'designer', 'economist', 'postmaster', 'broker', 'chemist', 'librarian', 'scientist', 'instructor', 'pilot', 'administrator', 'architect', 'surgeon', 'nurse', 'engineer', 'lawyer', 'physician', 'manager', 'official', 'doctor', 'professor', 'student', 'judge', 'teacher', 'author']
    # baseline_group = ['harris', 'nelson', 'robinson', 'thompson', 'moore', 'wright', 'anderson', 'clark', 'jackson', 'taylor', 'scott', 'davis', 'allen', 'adams', 'lewis', 'williams', 'jones', 'wilson', 'martin', 'johnson']
    # target_group = ['cho', 'wong', 'tang', 'huang', 'chu', 'chung', 'ng', 'wu', 'liu', 'chen', 'lin', 'yang', 'kim', 'chang', 'shah', 'wang', 'li', 'khan', 'singh', 'hong']

    ### Outsider vs all adjectives wrt Asians
    neutral_words = ['cho', 'wong', 'tang', 'huang', 'chu', 'chung', 'ng', 'wu', 'liu', 'chen', 'lin', 'yang', 'kim', 'chang', 'shah', 'wang', 'li', 'khan', 'singh', 'hong']
    baseline_group = ['headstrong', 'thankless', 'tactful', 'distrustful', 'quarrelsome', 'effeminate', 'fickle', 'talkative', 'dependable', 'resentful', 'sarcastic', 'unassuming', 'changeable', 'resourceful', 'persevering', 'forgiving', 'assertive', 'individualistic', 'vindictive', 'sophisticated', 'deceitful', 'impulsive', 'sociable', 'methodical', 'idealistic', 'thrifty', 'outgoing', 'intolerant', 'autocratic', 'conceited', 'inventive', 'dreamy', 'appreciative', 'forgetful', 'forceful', 'submissive', 'pessimistic', 'versatile', 'adaptable', 'reflective', 'inhibited', 'outspoken', 'quitting', 'unselfish', 'immature', 'painstaking', 'leisurely', 'infantile', 'sly', 'praising', 'cynical', 'irresponsible', 'arrogant', 'obliging', 'unkind', 'wary', 'greedy', 'obnoxious', 'irritable', 'discreet', 'frivolous', 'cowardly', 'rebellious', 'adventurous', 'enterprising', 'unscrupulous', 'poised', 'moody', 'unfriendly', 'optimistic', 'disorderly', 'peaceable', 'considerate', 'humorous', 'worrying', 'preoccupied', 'trusting', 'mischievous', 'robust', 'superstitious', 'noisy', 'tolerant', 'realistic', 'masculine', 'witty', 'informal', 'prejudiced', 'reckless', 'jolly', 'courageous', 'meek', 'stubborn', 'aloof', 'sentimental', 'complaining', 'unaffected', 'cooperative', 'unstable', 'feminine', 'timid', 'retiring', 'relaxed', 'imaginative', 'shrewd', 'conscientious', 'industrious', 'hasty', 'commonplace', 'lazy', 'gloomy', 'thoughtful', 'dignified', 'wholesome', 'affectionate', 'aggressive', 'awkward', 'energetic', 'tough', 'shy', 'queer', 'careless', 'restless', 'cautious', 'polished', 'tense', 'suspicious', 'dissatisfied', 'ingenious', 'fearful', 'daring', 'persistent', 'demanding', 'impatient', 'contented', 'selfish', 'rude', 'spontaneous', 'conventional', 'cheerful', 'enthusiastic', 'modest', 'ambitious', 'alert', 'defensive', 'mature', 'coarse', 'charming', 'clever', 'shallow', 'deliberate', 'stern', 'emotional', 'rigid', 'mild', 'cruel', 'artistic', 'hurried', 'sympathetic', 'dull', 'civilized', 'loyal', 'withdrawn', 'confident', 'indifferent', 'conservative', 'foolish', 'moderate', 'handsome', 'helpful', 'gentle', 'dominant', 'hostile', 'generous', 'reliable', 'sincere', 'precise', 'calm', 'healthy', 'attractive', 'progressive', 'confused', 'rational', 'stable', 'bitter', 'sensitive', 'initiative', 'loud', 'thorough', 'logical', 'intelligent', 'steady', 'formal', 'complicated', 'cool', 'curious', 'reserved', 'silent', 'honest', 'quick', 'friendly', 'efficient', 'pleasant', 'severe', 'peculiar', 'quiet', 'weak', 'anxious', 'nervous', 'warm', 'slow', 'dependent', 'wise', 'organized', 'affected', 'reasonable', 'capable', 'active', 'independent', 'patient', 'practical', 'serious', 'understanding', 'cold', 'responsible', 'simple', 'original', 'strong', 'determined', 'natural', 'kind']
    target_group = ['devious', 'bizarre', 'venomous', 'erratic', 'barbaric', 'frightening', 'deceitful', 'forceful', 'deceptive', 'envious', 'greedy', 'hateful', 'contemptible', 'brutal', 'monstrous', 'calculating', 'cruel', 'intolerant', 'aggressive', 'monstrous']

    ### Islam vs. Christianity wrt terrorism words
    # neutral_words = ['terror', 'terrorism', 'violence', 'attack', 'death', 'military', 'war', 'radical', 'injuries', 'bomb', 'target', 'conflict', 'dangerous', 'kill', 'murder', 'strike', 'dead', 'violence', 'fight', 'death', 'force', 'stronghold', 'wreckage', 'aggression', 'slaughter', 'execute', 'overthrow', 'casualties', 'massacre', 'retaliation', 'proliferation', 'militia', 'hostility', 'debris', 'acid', 'execution', 'militant', 'rocket', 'guerrilla', 'sacrifice', 'enemy', 'soldier', 'terrorist', 'missile', 'hostile', 'revolution', 'resistance', 'shoot']
    # baseline_group = ['baptism', 'messiah', 'catholicism', 'resurrection', 'christianity', 'salvation', 'protestant', 'gospel', 'trinity', 'jesus', 'christ', 'christian', 'cross', 'catholic', 'church']
    # target_group = ['allah', 'ramadan', 'turban', 'emir', 'salaam', 'sunni', 'koran', 'imam', 'sultan', 'prophet', 'veil', 'ayatollah', 'shiite', 'mosque', 'islam', 'sheik', 'muslim', 'muhammad']

    ### Refugee outsider vs. all adjectives
    # neutral_words = ['refugee', 'refugees', 'asylum', 'migrant', 'migrants', 'immigrant', 'immigrants']
    # baseline_group = ['headstrong', 'thankless', 'tactful', 'distrustful', 'quarrelsome', 'effeminate', 'fickle', 'talkative', 'dependable', 'resentful', 'sarcastic', 'unassuming', 'changeable', 'resourceful', 'persevering', 'forgiving', 'assertive', 'individualistic', 'vindictive', 'sophisticated', 'deceitful', 'impulsive', 'sociable', 'methodical', 'idealistic', 'thrifty', 'outgoing', 'intolerant', 'autocratic', 'conceited', 'inventive', 'dreamy', 'appreciative', 'forgetful', 'forceful', 'submissive', 'pessimistic', 'versatile', 'adaptable', 'reflective', 'inhibited', 'outspoken', 'quitting', 'unselfish', 'immature', 'painstaking', 'leisurely', 'infantile', 'sly', 'praising', 'cynical', 'irresponsible', 'arrogant', 'obliging', 'unkind', 'wary', 'greedy', 'obnoxious', 'irritable', 'discreet', 'frivolous', 'cowardly', 'rebellious', 'adventurous', 'enterprising', 'unscrupulous', 'poised', 'moody', 'unfriendly', 'optimistic', 'disorderly', 'peaceable', 'considerate', 'humorous', 'worrying', 'preoccupied', 'trusting', 'mischievous', 'robust', 'superstitious', 'noisy', 'tolerant', 'realistic', 'masculine', 'witty', 'informal', 'prejudiced', 'reckless', 'jolly', 'courageous', 'meek', 'stubborn', 'aloof', 'sentimental', 'complaining', 'unaffected', 'cooperative', 'unstable', 'feminine', 'timid', 'retiring', 'relaxed', 'imaginative', 'shrewd', 'conscientious', 'industrious', 'hasty', 'commonplace', 'lazy', 'gloomy', 'thoughtful', 'dignified', 'wholesome', 'affectionate', 'aggressive', 'awkward', 'energetic', 'tough', 'shy', 'queer', 'careless', 'restless', 'cautious', 'polished', 'tense', 'suspicious', 'dissatisfied', 'ingenious', 'fearful', 'daring', 'persistent', 'demanding', 'impatient', 'contented', 'selfish', 'rude', 'spontaneous', 'conventional', 'cheerful', 'enthusiastic', 'modest', 'ambitious', 'alert', 'defensive', 'mature', 'coarse', 'charming', 'clever', 'shallow', 'deliberate', 'stern', 'emotional', 'rigid', 'mild', 'cruel', 'artistic', 'hurried', 'sympathetic', 'dull', 'civilized', 'loyal', 'withdrawn', 'confident', 'indifferent', 'conservative', 'foolish', 'moderate', 'handsome', 'helpful', 'gentle', 'dominant', 'hostile', 'generous', 'reliable', 'sincere', 'precise', 'calm', 'healthy', 'attractive', 'progressive', 'confused', 'rational', 'stable', 'bitter', 'sensitive', 'initiative', 'loud', 'thorough', 'logical', 'intelligent', 'steady', 'formal', 'complicated', 'cool', 'curious', 'reserved', 'silent', 'honest', 'quick', 'friendly', 'efficient', 'pleasant', 'severe', 'peculiar', 'quiet', 'weak', 'anxious', 'nervous', 'warm', 'slow', 'dependent', 'wise', 'organized', 'affected', 'reasonable', 'capable', 'active', 'independent', 'patient', 'practical', 'serious', 'understanding', 'cold', 'responsible', 'simple', 'original', 'strong', 'determined', 'natural', 'kind']
    # target_group = ['illegal', 'illegals', 'devious', 'bizarre', 'venomous', 'erratic', 'barbaric', 'frightening', 'deceitful', 'forceful', 'deceptive', 'envious', 'greedy', 'hateful', 'contemptible', 'brutal', 'monstrous', 'calculating', 'cruel', 'intolerant', 'aggressive', 'monstrous', 'violent']

    # neutral_words = ['refugee']
    # baseline_group = ['kind']
    # target_group = ['illegals']

    # load model and data
    # dataset = load_dataset(data_file_name, model_ver)
    # num_docs = len(set([d.id for d in dataset]))
    vec_dim = int(re.search('_vecdim\.(\d+)_', model_file_name).group(1))
    num_docs = int(re.search('_numdocs\.(\d+)_', model_file_name).group(1))
    num_words = int(re.search('_vocabsize\.(\d+)_', model_file_name).group(1))

    # print 'loading data ...'
    # doc_ids, context_ids, target_noise_ids, word_to_ind_dict = load_and_cache_data(data_file_root=data_file_name)
    # all_doc_ids = set()
    # for i in doc_ids.tolist():
    #     all_doc_ids.add(i)

    print 'loading model ...'
    model, word_to_ind_dict = _load_model(
        model_file_name,
        vec_dim,
        num_docs=num_docs,
        num_words=num_words)

    # ind_to_word_dict = {}
    # for w in word_to_ind_dict:
    #     ind_to_word_dict[word_to_ind_dict[w]] = w

    # print (word_to_ind_dict['the'], ind_to_word_dict[0])
    # exit()

    if model_ver == 'dmspline':
        compute_bias_continuous(model, word_to_ind_dict, neutral_words, baseline_group, target_group)
    else:
        # compute_bias_categorical(model, word_to_ind_dict, num_docs, neutral_words, baseline_group, target_group)
        # check_average_similarities(model, word_to_ind_dict, num_docs)
        find_nearest_words(model, word_to_ind_dict, num_docs)


def find_nearest_words(model, word_to_ind_dict, num_docs):

    print ('Finding nearest words ...')

    w1 = 'war'

    all_words = word_to_ind_dict.keys()
    dist = defaultdict(list)
    count = 0
    for i in range(0, num_docs):
        word1_vec = np.array(model.get_baseline_word_vector(word_to_ind_dict[w1])) + np.array(model.get_paragraph_word_vector(i, word_to_ind_dict[w1]))
        # print ('baseline norm: ', np.linalg.norm(np.array(model.get_baseline_word_vector(word_to_ind_dict[w1]))))
        # print ('offset norm: ', np.linalg.norm( np.array(model.get_paragraph_word_vector(i, word_to_ind_dict[w1]))))
        # exit()
        # word1_vec = np.array(model.get_baseline_word_vector(word_to_ind_dict[w1]))
        for w2 in all_words:
            word2_vec = np.array(model.get_baseline_word_vector(word_to_ind_dict[w2])) + np.array(model.get_paragraph_word_vector(i, word_to_ind_dict[w2]))
            # word2_vec = np.array(model.get_baseline_word_vector(word_to_ind_dict[w2]))
            # print (w1, word1_vec)
            # print (w2, word2_vec)
            # exit()
            count += 1
            dist[w2].append(cosine(word1_vec, word2_vec))
            if count % 1000 == 0:
                print (count)

    for w in dist:
        dist[w] = np.mean(dist[w])
        # dist[w] = np.min(dist[w])

    print (json.dumps(sorted(dist.items(), key=lambda x:x[1], reverse=True), indent=4))


def check_average_similarities(model, word_to_ind_dict, num_docs):
    all_words = word_to_ind_dict.keys()
    word_1s = ['hate', 'school', 'random', 'good', 'great', 'war', 'black', 'he', 'england', 'soviet']
    word_2s = ['love', 'teacher', 'word', 'bad', 'great', 'gun', 'white', 'she', 'churchill', 'russia']

    # word_1s = ['hate', 'syrian', 'president', 'democrat', 'rush', 'fox', 'random', 'grocery', 'starbucks', 'illegal', 'hey']
    # word_2s = ['love', 'refugees', 'trump', 'republican', 'limbaugh', 'news', 'word', 'store', 'drink', 'immigrants', 'five']

    print ('vocab size: ', len(word_to_ind_dict))

    for w1, w2 in zip(word_1s, word_2s):

        if not w1 in word_to_ind_dict: 
            print ("skipping {}".format(w))
            continue

        if not w2 in word_to_ind_dict: 
            print ("skipping {}".format(w))
            continue

        baseline_wordvec_1 = np.array(model.get_baseline_word_vector(word_to_ind_dict[w1]))
        baseline_wordvec_2 = np.array(model.get_baseline_word_vector(word_to_ind_dict[w2]))
        print ('baseline cos dist for {} and {} is {}: '.format(w1, w2, cosine(baseline_wordvec_1, baseline_wordvec_2)))

        adjusted_sims = []
        for doc_ind in range(0, num_docs):

            word_1_offset = np.array(model.get_paragraph_word_vector(doc_ind, word_to_ind_dict[w1]))
            word_1_vec = baseline_wordvec_1 + word_1_offset
            word_1_vec /= np.linalg.norm(word_1_vec)

            word_2_offset = np.array(model.get_paragraph_word_vector(doc_ind, word_to_ind_dict[w2]))
            word_2_vec = baseline_wordvec_2 + word_2_offset
            word_2_vec /= np.linalg.norm(word_2_vec)

            # print (w1, baseline_wordvec_1, word_1_offset)
            # print (w2, baseline_wordvec_2, word_2_offset)
            # exit()
            adjusted_sims.append(cosine(word_1_vec, word_2_vec))

        print ('avg adjusted cosine dist for {} and {} is {}: '.format(w1, w2, np.mean(adjusted_sims)))


def compute_bias_categorical(model, word_to_ind_dict, num_docs, neutral_words, baseline_group, target_group):
    
    all_bias_scores = []

    # For each "document" (i.e., year), compute occupation gender bias 
    for i in range(0, num_docs):

        print 'Computing bias scores ...'
        bias, avg_baseline_norm, avg_offset_norm = _compute_relative_bias_score_categorical(model, word_to_ind_dict, i, neutral_words, baseline_group, target_group)
        all_bias_scores.append(bias)
        # print 'Bias for year %s is %s' % (1910 + i*10, bias)
        print 'Baseline norms: %s, offset norms: %s' % (avg_baseline_norm, avg_offset_norm)

    print all_bias_scores


def _compute_relative_bias_score_categorical(model, word_to_ind_dict, doc_ind, neutral_words, baseline_group, target_group):
    
    # Compute the average word distances for the baseline and refugee groups

    print 'Computing average vector for baseline words ...'
    mean_baseline_vec = _compute_average_vector_categorical(model, word_to_ind_dict, doc_ind, baseline_group)
    mean_baseline_vec = mean_baseline_vec / np.linalg.norm(mean_baseline_vec)

    print 'Computing average vector for target words ...'
    mean_target_vec = _compute_average_vector_categorical(model, word_to_ind_dict, doc_ind, target_group)
    mean_target_vec = mean_target_vec / np.linalg.norm(mean_target_vec)

    baseline_norms = []
    offset_norms = []

    # Now, for each neutral world, compute the difference in the relative norms wrt that word and average baseline and target vectors
    print 'Computing relative norm diff ...'
    bias = 0
    for w in neutral_words:

        if not w in word_to_ind_dict: 
            print ("skipping {}".format(w))
            continue

        baseline_vec = model.get_baseline_word_vector(word_to_ind_dict[w])
        vec_offset_for_doc = model.get_paragraph_word_vector(doc_ind, word_to_ind_dict[w])
        # neutral_wordvec = baseline_vec + vec_offset_for_doc
        neutral_wordvec = np.array(baseline_vec) + np.array(vec_offset_for_doc)
        neutral_wordvec = neutral_wordvec / np.linalg.norm(neutral_wordvec)

        baseline_norms.append(np.linalg.norm(baseline_vec))
        offset_norms.append(np.linalg.norm(vec_offset_for_doc))

        # bias += (np.linalg.norm(neutral_wordvec - mean_baseline_vec) - np.linalg.norm(neutral_wordvec - mean_target_vec))
        bias += cosine(neutral_wordvec, mean_baseline_vec) - cosine(neutral_wordvec, mean_target_vec)

    return bias, np.mean(baseline_norms), np.mean(offset_norms)


def _compute_average_vector_categorical(model, word_to_ind_dict, doc_ind, group):
    
    group_vecs = []
    for w in group:

        if not w in word_to_ind_dict: 
            print ("skipping {}".format(w))
            continue

        baseline_vec = model.get_baseline_word_vector(word_to_ind_dict[w])
        vec_offset_for_doc = model.get_paragraph_word_vector(doc_ind, word_to_ind_dict[w])
        # group_vecs.append(baseline_vec + vec_offset_for_doc)
        group_vecs.append(np.array(baseline_vec) + np.array(vec_offset_for_doc))

    # curr = np.mean(group_vecs)
    curr = np.mean(group_vecs, axis=0)
    return curr


def _load_model(model_file_name, vec_dim, num_docs, num_words):
    model_ver = re.search('_model\.(dmspline|dm|dbow)', model_file_name).group(1)
    if model_ver is None:
        raise ValueError("Model file name contains an invalid"
                         "version of the model")

    model_file_path = join(MODELS_DIR, model_file_name)

    try:
        checkpoint = torch.load(model_file_path)
    except AssertionError:
        checkpoint = torch.load(
            model_file_path,
            map_location=lambda storage, location: storage)

    if model_ver == 'dbow':
        model = DBOW(vec_dim, num_docs, num_words)
    elif model_ver == 'dm':
        model = DM(vec_dim, num_docs, num_words)
    else:
        model = DMSpline(vec_dim, num_docs, num_words)

    print 'LOADING MODEL: ', model_ver
    model.load_state_dict(checkpoint['model_state_dict'])
    # print model['word_to_index_dict']
    # exit()
    return model, checkpoint['word_to_index_dict']

if __name__ == '__main__':
    fire.Fire()
