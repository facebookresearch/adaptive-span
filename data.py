# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import os
import torch


class Dictionary(object):
    def __init__(self, path, sort_dict=False):
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = []

        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if sort_dict:
                        self.word2count[word] = self.word2count.get(word, 0) + 1
                    elif word not in self.word2idx:
                        self.word2idx[word] = len(self.idx2word)
                        self.idx2word.append(word)
        if sort_dict:
            # Sort dictionary by count and build indices accordingly:
            sorted_dict = sorted(self.word2count.items(), key=lambda kv: kv[1])[::-1]
            for i in range(len(sorted_dict)):
                word = sorted_dict[i][0]
                self.word2idx[word] = i
                self.idx2word.append(word)

    def __len__(self):
        return len(self.idx2word)


def _tokenize(text_path, dictionary):
    """Tokenizes a text file."""
    print('Tokenizing {}'.format(text_path))
    assert os.path.exists(text_path)

    # Assign to each token its identifier
    ids = []
    with open(text_path, 'r', encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ['<eos>']
            for token in tokens:
                ids.append(dictionary[token])
    ids = torch.LongTensor(ids)

    return ids


class Corpus:
    def __init__(self, data_path, sort_dict):
        print('Building dictionary')
        self._dictionary = Dictionary(os.path.join(data_path, 'train.txt'), sort_dict)

        self.train = _tokenize(
            text_path=os.path.join(data_path, 'train.txt'),
            dictionary=self._dictionary.word2idx)
        self.valid = _tokenize(
            text_path=os.path.join(data_path, 'valid.txt'),
            dictionary=self._dictionary.word2idx)
        self.test = _tokenize(
            text_path=os.path.join(data_path, 'test.txt'),
            dictionary=self._dictionary.word2idx)

    @property
    def vocab_size(self):
        return len(self._dictionary)


def _batchify(data_tensor, batch_size):
    nb_batches = data_tensor.size(0) // batch_size
    # trim away some tokens to make whole batches
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def _build_corpus(data_path, env_params, sort_dict):
    # save the corpus to a file so that it's faster next time
    if sort_dict:
        corpus_path = os.path.join(data_path, 'corpus_sorted.pt')
    else:
        corpus_path = os.path.join(data_path, 'corpus.pt')
    if os.path.exists(corpus_path):
        print('Loading an existing corpus file from {}'.format(corpus_path))
        corpus = torch.load(corpus_path)
    else:
        print('Creating a corpus file at {}'.format(corpus_path))
        if env_params['distributed']:
            # only one process need to create a corpus file
            if env_params['rank'] == 0:
                corpus = Corpus(data_path, sort_dict)
                torch.save(corpus, corpus_path)
                # sync with other processes
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
            else:
                print('Waiting rank0 to create a corpus file.')
                # sync with rank0
                torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                corpus = torch.load(corpus_path)
        else:
            corpus = Corpus(data_path, sort_dict)
            torch.save(corpus, corpus_path)
    return corpus


def _get_train_val_test_data(corpus, batch_size):
    return [
        _batchify(corpus.train, batch_size),
        _batchify(corpus.valid, batch_size),
        _batchify(corpus.test, batch_size)
    ]


def get_train_val_test_data(data_params, env_params, batch_size, device, sort_dict):
    corpus = _build_corpus(data_params['data_path'], env_params, sort_dict)
    data_params['vocab_size'] = corpus.vocab_size
    train_data, val_data, test_data = _get_train_val_test_data(
        corpus=corpus, batch_size=batch_size)

    if env_params['distributed']:
        # split the data into equal parts
        assert batch_size % env_params['world_size'] == 0
        device_batch_size = batch_size // env_params['world_size']
        slice_data = slice(
            device_batch_size * env_params['rank'],
            device_batch_size * (env_params['rank'] + 1))
        train_data = train_data[slice_data]
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    return train_data, val_data, test_data
