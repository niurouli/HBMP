import json
import numpy as np
from keras.utils import np_utils

def extract_tokens_from_binary_parse(parse):
    return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()


def yield_examples(fn, skip_no_majority=True, limit=None):
    for i, line in enumerate(open(fn)):
        if limit and i > limit:
            break
        data = json.loads(line)
        label = data['gold_label']
        s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
        s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
        if skip_no_majority and label == '-':
            continue
        yield (label, s1, s2)


def get_data(fn, limit=None):
    raw_data = list(yield_examples(fn=fn, limit=limit))
    left = [s1.strip().split() for _, s1, s2 in raw_data]
    right = [s2.strip().split() for _, s1, s2 in raw_data]


    LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
    Y = np_utils.to_categorical(Y, len(LABELS))

    return left, right, Y

def get_embdding_from_file(word_dict,filename):
    embeddings_index = {}
    f = open(filename, encoding='utf-8')
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    embd_dim = len(next(iter(embeddings_index.values())))
    weights = [[0.0] * embd_dim for _ in range(max(word_dict.values()) + 1)]
    for word, index in word_dict.items():
        if not word:
            continue
        word = word.lower()
        if word in embeddings_index:
            weights[index] = embeddings_index[word]
        else:
            weights[index] = np.random.random((embd_dim,)).tolist()
    word_embd_weights = [np.asarray(weights)]
    return word_embd_weights

