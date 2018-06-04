import os
import re
from itertools import islice, chain, cycle

from yxt_nlp_toolkit.common import Lang, Vocab
from yxt_nlp_toolkit.utils import tokenizer
from sklearn.model_selection import train_test_split

_dataset_path = os.path.expanduser("~/nlp/dataset/eng-fra.txt")

SOS = '<SOS>'
EOS = '<EOS>'
PAD = '<PAD>'
UNK = '<UNK>'


def _read_sentence_pair(max_seq_len, padding):
    padding = cycle([padding])
    with open(_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip(' \n')
            try:
                eng, fra = re.split('[\t]', line)
                eng_tokens = tuple(islice(
                    chain(tokenizer(eng, use_lib="spacy", lang="en"), padding),
                    0, max_seq_len))
                fra_tokens = tuple(islice(
                    chain(tokenizer(fra, use_lib="spacy", lang="fr"), padding),
                    0, max_seq_len))
                yield eng_tokens, fra_tokens
            except ValueError as e:
                print(line, e)


def _tokens(index, pair_seq):
    for one_pair in pair_seq:
        yield from one_pair[index]


def load_dataset(min_count, max_seq_len):
    pair_seq = list(islice(_read_sentence_pair(max_seq_len=max_seq_len, padding=PAD), 0, 10))
    en_vocab = Vocab(words=_tokens(0, pair_seq))
    fr_vocab = Vocab(words=_tokens(1, pair_seq))
    en_lang = Lang(words=en_vocab.shrink(min_count=min_count), reserved_tokens=(SOS, EOS, PAD), nil_token=UNK)
    fr_lang = Lang(words=fr_vocab.shrink(min_count=min_count), reserved_tokens=(SOS, EOS, PAD), nil_token=UNK)
    new_pairs = []
    for eng, fra in pair_seq:
        new_pairs.append((en_lang.to_indices(eng), fr_lang.to_indices(fra)))

    return en_lang, fr_lang, train_test_split(new_pairs, test_size=0.2, shuffle=True)

