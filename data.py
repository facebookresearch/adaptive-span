#!/usr/bin/env python3

import os
import torch


def tokenize(text_path, dictionary_to_update):
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
        self.train = tokenize(
            text_path=os.path.join(data_path, 'train.txt'),
            dictionary_to_update=self._dictionary)
        self.valid = tokenize(
            text_path=os.path.join(data_path, 'valid.txt'),
            dictionary_to_update=self._dictionary)
        self.test = tokenize(
            text_path=os.path.join(data_path, 'test.txt'),
            dictionary_to_update=self._dictionary)

    def __len__(self):
        return len(self._dictionary)


def batchify(data_tensor, batch_size):
    # Work out how cleanly we can divide the dataset into batch_size parts.
    nb_batches = data_tensor.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    # Evenly divide the data across the batch_size batches.
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def get_data(data_path,
             batch_size,
             world_size,
             device,
             distributed: bool):
    corpus_path = os.path.join(data_path, 'corpus.pt')
    if os.path.exists(corpus_path):
        corpus = torch.load(corpus_path)
    else:
        corpus = Corpus(data_path)
        torch.save(corpus, corpus_path)
    # TODO: see where vocab_sz is used
    args.vocab_sz = len(corpus)
    data = {
        'train_data': batchify(corpus.train, batch_size).to(device),
        'val_data': batchify(corpus.valid, batch_size).to(device),
        'test_data': batchify(corpus.test, batch_size).to(device)
    }
    if distributed:
        rank = torch.distributed.get_rank()
        torch.distributed.get_world_size()
        batch_sz = batch_size // world_size
        slice_data = slice(
            batch_sz * rank,
            batch_sz * (rank + 1))
    else:
        slice_data = slice(None)
    return {
        'train_data': batchify(corpus.train, batch_size).to(device)[slice_data],
        'val_data': batchify(corpus.valid, batch_size).to(device)[slice_data],
        'test_data': batchify(corpus.test, batch_size).to(device)[slice_data]
    }
