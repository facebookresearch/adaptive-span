#!/usr/bin/env python3

import os
import torch


def _tokenize(text_path, dictionary_to_update):
    """Tokenizes a text file."""
    assert os.path.exists(text_path)

    nb_tokens_in_dictionary = len(dictionary_to_update)
    nb_tokens_in_text = 0

    # Count nb of tokens in text and update the dictionary
    with open(text_path, 'r', encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ['<eos>']
            nb_tokens_in_text += len(tokens)
            for token in tokens:
                if token not in dictionary_to_update:
                    dictionary_to_update[token] = nb_tokens_in_dictionary
                    nb_tokens_in_dictionary += 1

    # Create tensor of size nb_tokens_in_text
    ids = torch.LongTensor(nb_tokens_in_text)
    # Assign to each token its identifier
    current_token_no = 0
    with open(text_path, 'r', encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ['<eos>']
            for token in tokens:
                ids[current_token_no] = dictionary_to_update[token]
                current_token_no += 1

    return ids


class Corpus:
    def __init__(self, data_path):
        self._dictionary = {}
        self.train = _tokenize(
            text_path=os.path.join(data_path, 'train.txt'),
            dictionary_to_update=self._dictionary)
        self.valid = _tokenize(
            text_path=os.path.join(data_path, 'valid.txt'),
            dictionary_to_update=self._dictionary)
        self.test = _tokenize(
            text_path=os.path.join(data_path, 'test.txt'),
            dictionary_to_update=self._dictionary)

    @property
    def vocab_size(self):
        return len(self._dictionary)


def _batchify(data_tensor, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nb_batches = data_tensor.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    # Evenly divide the data across the batch_size batches.
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def _build_corpus(data_path, *args, **kwargs):
    corpus_path = os.path.join(data_path, 'corpus.pt')
    if os.path.exists(corpus_path):
        corpus = torch.load(corpus_path)
    else:
        corpus = Corpus(data_path)
        torch.save(corpus, corpus_path)
    return corpus


def _get_train_val_test_data(corpus,
                             batch_size,
                             device,
                             rank: int,
                             *args,
                             **kwargs):
    # TODO: slice not compatible with tensor
    slice_data = slice(
        batch_size * rank,
        batch_size * (rank + 1))
    return [
        _batchify(corpus.train, batch_size).to(device)[slice_data],
        _batchify(corpus.valid, batch_size).to(device)[slice_data],
        _batchify(corpus.test, batch_size).to(device)[slice_data]
    ]


def get_train_val_test_data(data_params, env_params, device):
    corpus = _build_corpus(**data_params)
    data_params['vocab_size'] = corpus.vocab_size
    return _get_train_val_test_data(
        corpus=corpus, device=device, **env_params, **data_params)


def get_vocab_size(data_params):
    return data_params['vocab_size']
