import torch, re
from torchtext.vocab import Vocab
from torchtext.data import Field, TabularDataset

from utils import DATA_DIR, read_dict
from os.path import join

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


file_path = join(DATA_DIR, 'sample.csv')
id_field = Field(sequential=False, use_vocab=False, dtype=torch.int)
text_field = Field(pad_token=None, tokenize=_tokenize_str)

dataset = TabularDataset(
	path=file_path,
	format='csv',
	fields=[('id', id_field), ('text', text_field)],
	skip_header=True)

text_field.build_vocab(dataset)
curr_vocab = dataset.fields['text'].vocab
print curr_vocab.itos
print curr_vocab.stoi['the']