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


# Radio
# curr = {'migrants': Counter({u'16': 46, u'19': 39, u'10': 36, u'11': 34, u'14': 34, u'4': 32, u'18': 32, u'15': 30, u'31': 29, u'12': 25, u'13': 24, u'8': 22, u'30': 21, u'25': 18, u'22': 18, u'28': 18, u'20': 16, u'6': 16, u'24': 14, u'9': 14, u'3': 13, u'7': 13, u'26': 12, u'21': 12, u'23': 11, u'27': 10, u'0': 10, u'5': 10, u'17': 8, u'29': 7, u'2': 6, u'1': 4}), 'refugee': Counter({u'1': 63, u'0': 51, u'19': 43, u'6': 37, u'23': 35, u'2': 35, u'22': 34, u'13': 33, u'31': 33, u'8': 26, u'27': 24, u'3': 24, u'26': 23, u'4': 23, u'14': 22, u'5': 21, u'17': 21, u'10': 19, u'18': 19, u'30': 19, u'21': 17, u'29': 17, u'11': 17, u'12': 14, u'25': 12, u'7': 12, u'9': 11, u'15': 11, u'20': 10, u'16': 10, u'28': 9, u'24': 8}), 'immigrants': Counter({u'7': 382, u'8': 256, u'9': 160, u'22': 145, u'1': 138, u'6': 137, u'21': 131, u'12': 117, u'15': 114, u'23': 111, u'30': 111, u'2': 106, u'14': 101, u'0': 98, u'11': 91, u'16': 91, u'20': 87, u'5': 85, u'17': 82, u'10': 78, u'13': 75, u'3': 72, u'29': 68, u'31': 68, u'19': 62, u'4': 59, u'18': 40, u'28': 38, u'25': 37, u'26': 36, u'24': 32, u'27': 32}), 'immigrant': Counter({u'7': 375, u'8': 243, u'6': 116, u'9': 95, u'22': 93, u'21': 78, u'23': 76, u'10': 72, u'16': 65, u'12': 64, u'15': 62, u'17': 61, u'5': 59, u'1': 52, u'30': 51, u'19': 50, u'11': 46, u'13': 45, u'20': 38, u'3': 37, u'0': 36, u'18': 36, u'4': 34, u'31': 33, u'26': 31, u'2': 30, u'25': 28, u'29': 27, u'14': 26, u'24': 25, u'28': 23, u'27': 21}), 'migrant': Counter({u'2': 30, u'18': 28, u'19': 22, u'22': 20, u'23': 20, u'16': 20, u'4': 18, u'1': 17, u'6': 14, u'14': 14, u'30': 14, u'3': 12, u'5': 12, u'8': 12, u'31': 12, u'28': 11, u'0': 11, u'11': 10, u'10': 10, u'26': 8, u'29': 8, u'7': 8, u'13': 8, u'15': 8, u'25': 7, u'27': 6, u'17': 6, u'21': 5, u'9': 5, u'12': 4, u'20': 3, u'24': 1}), 'asylum': Counter({u'31': 73, u'6': 56, u'23': 45, u'21': 43, u'3': 43, u'22': 39, u'29': 31, u'20': 30, u'27': 28, u'30': 23, u'10': 21, u'2': 19, u'18': 18, u'0': 16, u'9': 16, u'24': 15, u'25': 15, u'13': 14, u'4': 13, u'14': 13, u'19': 13, u'26': 11, u'16': 11, u'5': 10, u'11': 10, u'1': 8, u'7': 8, u'8': 8, u'12': 7, u'17': 7, u'28': 4, u'15': 1}), 'refugees': Counter({u'17': 78, u'13': 70, u'6': 56, u'22': 44, u'23': 43, u'8': 41, u'31': 41, u'1': 40, u'16': 40, u'10': 38, u'2': 37, u'26': 33, u'4': 32, u'0': 30, u'14': 29, u'29': 26, u'11': 26, u'19': 26, u'18': 26, u'3': 22, u'7': 22, u'12': 20, u'27': 19, u'5': 19, u'28': 18, u'9': 18, u'25': 16, u'20': 16, u'24': 14, u'30': 12, u'21': 10, u'15': 10})}

# COHA


# Counting word occurrences per doc
from scipy.stats import pearsonr
bias = [-1.8683822386395362e-05, 8.057661843262061e-08, 2.4109535211062016e-07, 4.187659111485435e-06, 8.693319380791365e-07, 4.792908133319132e-06, -2.0969056647196396e-07, 8.172500282396838e-06, 3.44198182798569e-05]
total_counts = [0 for i in range(0, len(bias))]
for w in curr:
    a = [0 for i in range(0, len(bias))]
    for ind in curr[w]:
        a[int(ind)] = curr[w][ind]
        total_counts[int(ind)] += curr[w][ind]
    print (w, pearsonr(a,bias), np.sum(a))    

print ('all', pearsonr(total_counts, bias))

