import multiprocessing
import os
import pickle
import re
import string
import signal
from math import ceil
from os.path import join

import numpy as np
import datetime
import json
import pandas as pd
import torch
from numpy.random import choice
from torchtext.data import Field, TabularDataset
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler

from collections import defaultdict
from utils import DATA_DIR, read_dict

import sys
import csv

csv.field_size_limit(sys.maxsize)

NUM = re.compile("^\d*[-\./,]*\d+$")
PUNCT = set(string.punctuation)
PUNCT.add('--')
AFFIX = set(["n't", "'s", "'d", "'t"]) 

def clean_text(text):
    curr = text.lower()
    curr = curr.replace('\0', '')

    curr = curr.split(' ')
    to_return = []
    for i in range(0, len(curr)):
        word = curr[i]
        if word == "@" or word =="<p>":
            continue
        elif word in PUNCT:
            continue
        elif word in AFFIX:
            continue
        else:
            word = word.strip().strip("*").lower()
            if NUM.match(word):
                word = "<NUM>"
            to_return.append(word)

    curr = ' '.join(to_return)
    curr = curr.encode('utf-8')
    return curr

def output_nyt_data_in_format(
        data_dir='/Users/ngillani/data/nyt_corpus/data/',
        start_year=1987,
        end_year=2006,
        output_file='data/nyt_data_%s_through_%s_sampled.csv',
        num_docs=20

    ):

    data_by_year = defaultdict(list)
    from bs4 import BeautifulSoup
    all_data = {'text': [], 'id': []}
    for y in range(start_year, end_year + 1):
        y_int = y
        y = str(y)
        for m in os.listdir(os.path.join(data_dir, y)):
            for d in os.listdir(os.path.join(data_dir, y, m)):
                all_files = os.listdir(os.path.join(data_dir, y, m, d))
                np.random.shuffle(all_files)
                subset_files = all_files[0:num_docs]
                for a in subset_files:
                    # if i == 10: break

                    try:

                        soup = BeautifulSoup(open(os.path.join(data_dir, y, m, d, a)).read(), 'lxml')
                        curr = soup.find('block', attrs={'class': 'full_text'})
                        curr = curr.findAll('p')

                        # Drop the lead paragraph since this is incorporated into the subsequent paragraphs
                        curr.pop(0)

                        all_text = [p.get_text() for p in curr]
                        curr_doc = []
                        for t in all_text:
                            try:
                                t = t.encode('utf-8').lower()
                                curr_doc.append(t)
                                # data_by_year[y].append(t)
                            except Exception as e:
                                print e
                                continue
                        text_to_add = ' '.join(curr_doc)

                        if y_int == start_year:
                            all_data['id'].append(y_int - start_year)
                            all_data['text'].append(text_to_add)
                        elif y_int == end_year:
                            all_data['id'].append((y_int - 2) - start_year)
                            all_data['text'].append(text_to_add)
                        elif y_int == start_year + 1:
                            all_data['id'].append((y_int - 1) - start_year)
                            all_data['text'].append(text_to_add)
                            all_data['id'].append(y_int - start_year)
                            all_data['text'].append(text_to_add)
                        elif y_int == end_year - 1:
                            all_data['id'].append((y_int - 2) - start_year)
                            all_data['text'].append(text_to_add)
                            all_data['id'].append((y_int - 1) - start_year)
                            all_data['text'].append(text_to_add)
                        else:
                            all_data['id'].append((y_int - 2) - start_year)
                            all_data['text'].append(text_to_add)
                            all_data['id'].append((y_int - 1) - start_year)
                            all_data['text'].append(text_to_add)
                            all_data['id'].append(y_int - start_year)
                            all_data['text'].append(text_to_add)

                    except Exception as e:
                        print e
                        continue

    df = pd.DataFrame(data=all_data)
    df.to_csv(output_file % (start_year + 1, end_year - 1), index=False)


def output_coha_data_in_format(
        data_dir='/Users/ngillani/data/coha/',
        start_decade=1910,
        end_decade=2000,
        output_file='data/all_data_%s_through_%s.csv'
    ):

    step = 10
    
    all_data = {'id': [], 'text': []}
    total_data = []
    for d in range(start_decade, end_decade, step):

        all_text_to_add = []
        decade_dir = str(d)+'s/'
        count = 0
        curr_files = os.listdir(data_dir+decade_dir)
        np.random.shuffle(curr_files)
        for i, f in enumerate(curr_files):
            curr = open(data_dir + decade_dir + f)
            curr.readline()
            curr.readline()
            try:
                curr_text = clean_text(curr.read())
                all_text_to_add.append(curr_text)
                all_data['text'].append(curr_text)
                all_data['id'].append(int((d - start_decade)/step))
            except Exception as e:
                print e
                continue

        data_for_decade = ' '.join(all_text_to_add)
        print 'Num files for %s: %s --- size: %s MB' % (d, i, np.round(float(sys.getsizeof(data_for_decade))/1000000,2))

    df = pd.DataFrame(data=all_data)
    df.to_csv(output_file % (start_decade, end_decade - 10), index=False)


def output_sampled_data(
        input_file='data/all_data_1910_through_1990.csv',
        output_file='data/half_all_data_1910_through_1990.csv',
        prop=0.5
    ):

    print ('Loading data ...')
    df = pd.read_csv(input_file)
    df_s = df.sample(frac=prop, replace=False)
    df_s.to_csv(output_file, index=False)


def output_radio_data_by_geo(
        data_file='/Users/ngillani/data/radio/2018_09_single_callsign_show_data.json',
        output_file='data/radio_data_by_city_and_state.csv',
        output_mapping_file='data/radio_by_city_and_state_mapping.json'
    ):

    radio_data = read_dict(data_file)
    cities_to_ids = {}

    all_data = {'id': [], 'text': []}

    for i, s in enumerate(radio_data):
        print i
        geo = s['city'] + ', ' + s['state']
        if not geo in cities_to_ids:
            cities_to_ids[geo] = len(cities_to_ids)

        all_data['id'].append(cities_to_ids[geo])
        all_data['text'].append(s['denorm_content'])

    df = pd.DataFrame(data=all_data)
    df.to_csv(output_file, index=False)

    f = open(output_mapping_file, 'w')
    f.write(json.dumps(cities_to_ids, indent=4))
    f.close()


def output_radio_data_by_day(
        data_file='/Users/ngillani/data/radio/2018_09_single_callsign_show_data.json',
        output_file='data/radio_data_by_day.csv'
    ):

    radio_data = read_dict(data_file)
    cities_to_ids = {}

    all_data = {'id': [], 'text': []}
    ref_date = datetime.date(2018, 9, 1)

    all_days = set()
    for i, s in enumerate(radio_data):
        print i
        # if i == 100: break
        curr_date = datetime.datetime.utcfromtimestamp(s['segment_start_global']).date()

        day_id = (curr_date - ref_date).days
        all_days.add(day_id)
        all_data['id'].append(day_id)
        all_data['text'].append(s['denorm_content'])

    print all_days

    df = pd.DataFrame(data=all_data)
    df.to_csv(output_file, index=False)


def output_radio_data_by_day_multi_month(
        data_file_1='/Users/ngillani/data/radio/2018_08_single_callsign_show_data.json',
        data_file_2='/Users/ngillani/data/radio/2018_09_single_callsign_show_data.json',
        start_date=datetime.date(2018, 8, 15),
        end_date=datetime.date(2018, 9, 15),
        output_file='data/radio_data_by_day_mid_aug_mid_sept.csv'
    ):

    radio_data = [read_dict(data_file_1), read_dict(data_file_2)]
    cities_to_ids = {}

    all_data = {'id': [], 'text': []}
    ref_date = datetime.date(2018, 8, 15)

    all_days = set()
    for j in range(0, len(radio_data)):
        for i, s in enumerate(radio_data[j]):
            print j, i
            # if i == 100: break
            curr_date = datetime.datetime.utcfromtimestamp(s['segment_start_global']).date()

            if curr_date < start_date or curr_date > end_date: continue

            day_id = (curr_date - ref_date).days
            all_days.add(day_id)
            all_data['id'].append(day_id)
            all_data['text'].append(s['denorm_content'])

    print all_days

    df = pd.DataFrame(data=all_data)
    df.to_csv(output_file, index=False)


def get_knot_id(knots, attr):
    for i in range(0, len(knots)):
        if attr <= knots[i]:
            # print 'returning %s for attr %s' % (i, attr)
            return i

    # print knots
    # print attr
    raise Exception('Error in finding knot ID')


def output_radio_data_by_day_multi_month_for_continuous_context(
        data_file_1='/Users/ngillani/data/radio/2018_08_single_callsign_show_data.json',
        data_file_2='/Users/ngillani/data/radio/2018_09_single_callsign_show_data.json',
        knot_size=3,
        start_date=datetime.date(2018, 8, 15),
        end_date=datetime.date(2018, 9, 15),
        output_file='data/radio_data_by_day_mid_aug_mid_sept_continuous.csv'
    ):

    radio_data = [read_dict(data_file_1), read_dict(data_file_2)]
    # radio_data = [read_dict(data_file_1)]
    cities_to_ids = {}

    all_data = {'id': [], 'attr': [], 'standardized_attr': [], 'text': []}
    ref_date = datetime.date(2018, 8, 15)

    all_days = set()
    for j in range(0, len(radio_data)):
        for i, s in enumerate(radio_data[j]):
            print j, i
            # if i == 10000: break
            curr_date = datetime.datetime.utcfromtimestamp(s['segment_start_global']).date()

            if curr_date < start_date or curr_date > end_date: continue

            day_id = (curr_date - ref_date).days
            all_days.add(day_id)
            all_data['attr'].append(day_id)
            all_data['text'].append(s['denorm_content'])

    all_days = list(all_days)
    all_days_mean = np.mean(all_days)
    all_days_std = np.std(all_days)
    knots = list(range(np.min(all_days), np.max(all_days) + knot_size, knot_size))
    for d in all_data['attr']:
        all_data['standardized_attr'].append(float(d - all_days_mean) / all_days_std)
        all_data['id'].append(get_knot_id(knots, d))

    df = pd.DataFrame(data=all_data)
    df.to_csv(output_file, index=False)
    print knots


def load_dataset(file_name, model_ver):
    """Loads contents from a file in the *data* directory into a
    torchtext.data.TabularDataset instance.
    """

    print 'Loading dataset ...'

    if model_ver == 'dmspline':
        file_path = join(DATA_DIR, file_name)
        id_field = Field(sequential=False, use_vocab=False, dtype=torch.int)
        text_field = Field(pad_token=None, tokenize=_tokenize_str)
        attr_field = Field(sequential=False, use_vocab=False, dtype=torch.float)
        standardzed_attr_field = Field(sequential=False, use_vocab=False, dtype=torch.float)

        dataset = TabularDataset(
            path=file_path,
            format='csv',
            fields=[('attr', attr_field), ('id', id_field), ('standardized_attr', standardzed_attr_field), ('text', text_field)],
            skip_header=True)

    else:
        file_path = join(DATA_DIR, file_name)
        id_field = Field(sequential=False, use_vocab=False, dtype=torch.int)
        text_field = Field(pad_token=None, tokenize=_tokenize_str)

        dataset = TabularDataset(
            path=file_path,
            format='csv',
            fields=[('id', id_field), ('text', text_field)],
            skip_header=True)

    text_field.build_vocab(dataset, min_freq=10)
    return dataset



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


class NCEData(object):
    """An infinite, parallel (multiprocess) batch generator for
    noise-contrastive estimation of word vector models.

    Parameters
    ----------
    dataset: torchtext.data.TabularDataset
        Dataset from which examples are generated. A column labeled *text*
        is expected and should be comprised of a list of tokens. Each row
        should represent a single document.

    batch_size: int
        Number of examples per single gradient update.

    context_size: int
        Half the size of a neighbourhood of target words (i.e. how many
        words left and right are regarded as context).

    num_noise_words: int
        Number of noise words to sample from the noise distribution.

    max_size: int
        Maximum number of pre-generated batches.

    num_workers: int
        Number of jobs to run in parallel. If value is set to -1, total number
        of machine CPUs is used.
    """
    # code inspired by parallel generators in https://github.com/fchollet/keras
    def __init__(self, dataset, batch_size, context_size,
                 num_noise_words, max_size, num_workers, model_ver):
        self.max_size = max_size

        self.num_workers = num_workers if num_workers != -1 else os.cpu_count()
        if self.num_workers is None:
            self.num_workers = 1

        self._generator = _NCEGenerator(
            dataset,
            batch_size,
            context_size,
            num_noise_words,
            model_ver,
            _NCEGeneratorState(context_size))

        self._queue = None
        self._stop_event = None
        self._processes = []

    def __len__(self):
        return len(self._generator)

    def vocabulary_size(self):
        return self._generator.vocabulary_size()

    def start(self):
        """Starts num_worker processes that generate batches of data."""
        self._queue = multiprocessing.Queue(maxsize=self.max_size)
        self._stop_event = multiprocessing.Event()

        for _ in range(self.num_workers):
            process = multiprocessing.Process(target=self._parallel_task)
            process.daemon = True
            self._processes.append(process)
            process.start()

    def _parallel_task(self):
        while not self._stop_event.is_set():
            try:
                batch = self._generator.next()
                # queue blocks a call to put() until a free slot is available
                self._queue.put(batch)
            except KeyboardInterrupt:
                self._stop_event.set()

    def get_generator(self):
        """Returns a generator that yields batches of data."""
        while self._is_running():
            yield self._queue.get()

    def stop(self):
        """Terminates all processes that were created with start()."""
        if self._is_running():
            self._stop_event.set()

        for process in self._processes:
            if process.is_alive():
                os.kill(process.pid, signal.SIGINT)
                process.join()

        if self._queue is not None:
            self._queue.close()

        self._queue = None
        self._stop_event = None
        self._processes = []

    def _is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()


class _NCEGenerator(object):
    """An infinite, process-safe batch generator for noise-contrastive
    estimation of word vector models.

    Parameters
    ----------
    state: paragraphvec.data._NCEGeneratorState
        Initial (indexing) state of the generator.

    For other parameters see the NCEData class.
    """
    def __init__(self, dataset, batch_size, context_size,
                 num_noise_words, model_ver, state):
        self.dataset = dataset
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_noise_words = num_noise_words

        self._vocabulary = self.dataset.fields['text'].vocab
        self._sample_noise = None
        self._init_noise_distribution()
        self._model_ver = model_ver
        self._state = state

    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        probs = np.zeros(len(self._vocabulary))

        for word, freq in self._vocabulary.freqs.items():
            probs[self._word_to_index(word)] = freq

        probs = np.power(probs, 0.75)
        probs /= np.sum(probs)

        self._sample_noise = lambda: choice(
            probs.shape[0], self.num_noise_words, p=probs).tolist()

    def __len__(self):
        num_examples = sum(self._num_examples_in_doc(d) for d in self.dataset)
        return ceil(num_examples / self.batch_size)

    def vocabulary_size(self):
        return len(self._vocabulary)

    def next(self):
        """Updates state for the next process in a process-safe manner
        and generates the current batch."""
        prev_doc_id, prev_in_doc_pos = self._state.update_state(
            self.dataset,
            self.batch_size,
            self.context_size,
            self._num_examples_in_doc)

        # generate the actual batch
        batch = _NCEBatch(self.context_size)

        while len(batch) < self.batch_size:
            if prev_doc_id == len(self.dataset):
                # last document exhausted
                batch.torch_()
                return batch
            if prev_in_doc_pos <= (len(self.dataset[prev_doc_id].text) - 1
                                   - self.context_size):
                # more examples in the current document
                self._add_example_to_batch(prev_doc_id, prev_in_doc_pos, batch)
                prev_in_doc_pos += 1
            else:
                # go to the next document
                prev_doc_id += 1
                prev_in_doc_pos = self.context_size

        batch.torch_()
        return batch

    def _num_examples_in_doc(self, doc, in_doc_pos=None):
        if in_doc_pos is not None:
            # number of remaining
            if len(doc.text) - in_doc_pos >= self.context_size + 1:
                return len(doc.text) - in_doc_pos - self.context_size
            return 0

        if len(doc.text) >= 2 * self.context_size + 1:
            # total number
            return len(doc.text) - 2 * self.context_size
        return 0

    def _add_example_to_batch(self, doc_id, in_doc_pos, batch):
        doc = self.dataset[doc_id].text
        batch.doc_ids.append(int(self.dataset[doc_id].id))

        if self._model_ver == 'dmspline':
            batch.raw_x_vals.append(float(self.dataset[doc_id].attr))
            batch.x_vals.append(float(self.dataset[doc_id].standardized_attr))

        # sample from the noise distribution
        current_noise = self._sample_noise()
        current_noise.insert(0, self._word_to_index(doc[in_doc_pos]))
        batch.target_noise_ids.append(current_noise)

        if self.context_size == 0:
            return

        current_context = []
        context_indices = (in_doc_pos + diff for diff in
                           range(-self.context_size, self.context_size + 1)
                           if diff != 0)

        current_words = []
        for i in context_indices:
            # print ('doc id: ', int(self.dataset[doc_id].id), 'ind: ', i, 'word: ', doc[i])
            context_id = self._word_to_index(doc[i])
            current_context.append(context_id)
            current_words.append(doc[i])
        batch.context_ids.append(current_context)
        batch.context_words.append(current_words)
    def _word_to_index(self, word):
        return self._vocabulary.stoi[word]


class _NCEGeneratorState(object):
    """Batch generator state that is represented with a document id and
    in-document position. It abstracts a process-safe indexing mechanism."""
    def __init__(self, context_size):
        # use raw values because both indices have
        # to manually be locked together
        self._doc_id = multiprocessing.RawValue('i', 0)
        self._in_doc_pos = multiprocessing.RawValue('i', context_size)
        self._lock = multiprocessing.Lock()

    def update_state(self, dataset, batch_size,
                     context_size, num_examples_in_doc):
        """Returns current indices and computes new indices for the
        next process."""
        with self._lock:
            doc_id = self._doc_id.value
            in_doc_pos = self._in_doc_pos.value
            self._advance_indices(
                dataset, batch_size, context_size, num_examples_in_doc)
            return doc_id, in_doc_pos

    def _advance_indices(self, dataset, batch_size,
                         context_size, num_examples_in_doc):
        num_examples = num_examples_in_doc(
            dataset[self._doc_id.value], self._in_doc_pos.value)

        if num_examples > batch_size:
            # more examples in the current document
            self._in_doc_pos.value += batch_size
            return

        if num_examples == batch_size:
            # just enough examples in the current document
            if self._doc_id.value < len(dataset) - 1:
                self._doc_id.value += 1
            else:
                self._doc_id.value = 0
            self._in_doc_pos.value = context_size
            return

        while num_examples < batch_size:
            if self._doc_id.value == len(dataset) - 1:
                # last document: reset indices
                self._doc_id.value = 0
                self._in_doc_pos.value = context_size
                return

            self._doc_id.value += 1
            num_examples += num_examples_in_doc(
                dataset[self._doc_id.value])

        self._in_doc_pos.value = (len(dataset[self._doc_id.value].text)
                                  - context_size
                                  - (num_examples - batch_size))


class _NCEBatch(object):
    def __init__(self, context_size):
        self.context_ids = [] if context_size > 0 else None
        self.context_words = [] if context_size > 0 else None
        self.doc_ids = []
        self.target_noise_ids = []
        self.raw_x_vals = []
        self.x_vals = []

    def __len__(self):
        return len(self.doc_ids)

    def torch_(self):
        if self.context_ids is not None:
            self.context_ids = torch.LongTensor(self.context_ids)
        self.doc_ids = torch.LongTensor(self.doc_ids)
        self.target_noise_ids = torch.LongTensor(self.target_noise_ids)
        self.raw_x_vals = torch.FloatTensor(self.raw_x_vals)
        self.x_vals = torch.FloatTensor(self.x_vals)

    def cuda_(self):
        if self.context_ids is not None:
            self.context_ids = self.context_ids.cuda()
        self.doc_ids = self.doc_ids.cuda()
        self.target_noise_ids = self.target_noise_ids.cuda()
        self.raw_x_vals = self.raw_x_vals.cuda()
        self.x_vals = self.x_vals.cuda()


if __name__ == "__main__":
    # output_nyt_data_in_format()
    # output_coha_data_in_format()
    # output_radio_data_by_geo()
    # output_radio_data_by_day()
    # output_radio_data_by_day_multi_month()
    # output_radio_data_by_day_multi_month_for_continuous_context()
    output_sampled_data()


