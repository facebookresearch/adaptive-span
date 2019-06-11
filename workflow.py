#!/usr/bin/env python3

import os
import math
import time

import torch

# TODO: review import statements
from models import TransformerSeq
from adagrad_with_grad_clip import AdagradWithGradClip
from utils import (
    load_checkpoint,
    is_checkpoint,
    save_checkpoint,
    Logger,
    Plotter)


##############################################################################
# ENVIRONMENT
##############################################################################

def _torch_distributed_init_process_group(distributed,
                                          submitit_enabled,
                                          dist_init,
                                          local_rank):
    rank, world_size = 0, 1
    if distributed:
        if submitit_enabled:
            # remove
            import submitit
            job_env = submitit.JobEnvironment()
            rank = job_env.global_rank
            world_size = job_env.num_tasks
            local_rank = job_env.local_rank
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=dist_init,
                rank=rank,
                world_size=world_size
            )
        else:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        torch.cuda.set_device(local_rank)
    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
    }


def _get_device(cuda_enabled):
    if cuda_enabled and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_up_env(env_params):
    env_params.update(
        _torch_distributed_init_process_group(
            distributed=env_params['distributed'],
            submitit_enabled=env_params['submitit_enabled'],
            dist_init=env_params['dist_init'],
            local_rank=env_params['local_rank']))
    env_params['device'] = _get_device(
        cuda_enabled=not env_params['no_cuda'])


def get_device(env_params):
    return env_params['device']


##############################################################################
# DATA
##############################################################################

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
            print(nb_tokens_in_text)
            for token in tokens:
                if token not in dictionary_to_update:
                    dictionary_to_update[token] = nb_tokens_in_dictionary
                    nb_tokens_in_dictionary += 1
    breakpoint()

    # Create tensor of size nb_tokens_in_text
    ids = torch.LongTensor(nb_tokens_in_text)
    # Assign to each token its identifier
    current_token_no = 0
    with open(text_path, 'r', encoding="utf8") as f:
        old_percentage = -1
        for line in f:
            tokens = line.split() + ['<eos>']
            for token in tokens:
                ids[current_token_no] = dictionary_to_update[token]
                current_token_no += 1
                percentage = ((10000 * current_token_no) // nb_tokens_in_text) / 100
                if percentage > old_percentage:
                    old_percentage = percentage
                    print(f'{old_percentage}% '
                          f'{current_token_no} / {nb_tokens_in_text} tokens',
                          end='\r')
    print('')
    breakpoint()

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


def _build_corpus(data_path):
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
                             rank):
    slice_data = slice(
        batch_size * rank,
        batch_size * (rank + 1))
    return [
        _batchify(corpus.train, batch_size).to(device)[slice_data],
        _batchify(corpus.valid, batch_size).to(device)[slice_data],
        _batchify(corpus.test, batch_size).to(device)[slice_data]
    ]


def get_train_val_test_data(data_params, env_params, optim_params, device):
    corpus = _build_corpus(**data_params)
    data_params['vocab_size'] = corpus.vocab_size
    return _get_train_val_test_data(corpus=corpus,
                                    device=device,
                                    batch_size=optim_params['batch_size'],
                                    rank=env_params['rank'])


def get_vocab_size(data_params):
    return data_params['vocab_size']


##############################################################################
# MODEL
##############################################################################

def _get_model(device,
               vocab_size,
               local_rank,
               distributed,
               model_params,
               attn_span_params):
    model = TransformerSeq(
        vocab_size=vocab_size,
        hidden_size=model_params['hidden_size'],
        nb_heads=model_params['nb_heads'],
        nb_layers=model_params['nb_layers'],
        attn_span_lim=attn_span_params['attn_span_lim'],
        block_size=model_params['block_size'],
        model_params=model_params,
        attn_span_params=attn_span_params)
    if distributed:
        model = model.to(device, dtype=torch.float32)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device, dtype=torch.float32)
    return model


def get_model(model_params, attn_span_params, env_params, device, vocab_size):
    return _get_model(device=device,
                      vocab_size=vocab_size,
                      model_params=model_params,
                      attn_span_params=attn_span_params,
                      local_rank=env_params['local_rank'],
                      distributed=env_params['distributed'])


##############################################################################
# OPTIMIZER AND SCHEDULER
##############################################################################

def _get_grad_requiring_params(model):
    nb_parameters = 0
    grad_requiring_params = []
    for param in model.parameters():
        if param.requires_grad:
            nb_parameters += param.numel()
            grad_requiring_params.append(param)
    print('nb_parameters={:.2f}M'.format(nb_parameters / 1e6))
    return grad_requiring_params


def _get_optimizer(model,
                   optim,
                   lr: float,
                   momentum: float,
                   grad_clip: float):
    if optim == 'sgd':
        return torch.optim.SGD(_get_grad_requiring_params(model),
                               lr=lr,
                               momentum=momentum)
    elif optim == 'adagrad':
        return AdagradWithGradClip(_get_grad_requiring_params(model),
                                   lr=lr,
                                   grad_clip=grad_clip)
    else:
        raise RuntimeError("wrong type of optimizer "
                           "- must be 'sgd' or 'adagrad")


def _get_scheduler(optimizer, lr_warmup):
    if lr_warmup > 0:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda ep: min(1, ep / lr_warmup))
    return None


def update_optim_params(optim_params, env_params):
    optim_params['batch_size'] //= env_params['world_size']


def get_optimizer_and_scheduler(model, optim_params):
    optimizer = _get_optimizer(model=model,
                               optim=optim_params['optim'],
                               lr=optim_params['lr'],
                               momentum=optim_params['momentum'],
                               grad_clip=optim_params['grad_clip'])
    scheduler = _get_scheduler(optimizer=optimizer,
                               lr_warmup=optim_params['lr_warmup'])
    return optimizer, scheduler


##############################################################################
# TRAINER
##############################################################################

def _log_iter(logger,
              iter_no,
              nb_batches,
              stat_train,
              stat_val,
              elapsed,
              attn_span_loss,
              model):
    X = (iter_no + 1) * nb_batches
    # TODO: why log(2)
    train_bpc = stat_train['loss'] / math.log(2)
    val_bpc = stat_val['loss'] / math.log(2)
    print('{}\ttrain: {:.2f}bpc\tval: {:.2f}bpc\tms/batch: {:.1f}'.format(
        X, train_bpc, val_bpc, elapsed))
    logger.log(title='X', value=X)
    logger.log(title='train_bpc', value=train_bpc)
    logger.log(title='val_bpc', value=val_bpc)

    span_latest = []
    if attn_span_loss > 0:
        for layer in model.module.layers:
            span = layer.attn.attn.adaptive_span.mask.size_ratio.view(-1)
            span_latest.append(span)
            # TODO: why this line?
            span = span.mean().item()
        span_latest = torch.cat(span_latest, dim=0)
        logger.log('span_avg', span_latest.mean().item())
        logger.log('span_max', span_latest.max().item())
    return span_latest


def _plot_iter(plotter, span_latest, logger):
    plotter.plot(title='train_bpc',
                 X=logger.get_data('X'),
                 Y=logger.get_data('train_bpc'))
    plotter.plot(title='val_bpc',
                 X=logger.get_data('X'),
                 Y=logger.get_data('val_bpc'))
    if span_latest:
        plotter.plot(title='span_latest',
                     Y=logger.get_data('span_latest'))
    plotter.save()


def _save_iter(checkpoint_freq,
               checkpoint_path,
               iter_no,
               model,
               optimizer,
               scheduler,
               logger):
    actual_checkpoint_path = checkpoint_path
    if is_checkpoint(iter_no, checkpoint_freq):
        actual_checkpoint_path += f".{iter_no+1}"
    save_checkpoint(
        checkpoint_path=actual_checkpoint_path,
        iter_no=iter_no,
        model=model,
        optimizer=optimizer,
        logger=logger,
        scheduler=scheduler)


# separating batch training reduces memory usage (removes overlap?)
def _train_batch(model,
                 optimizer,
                 scheduler,
                 data,
                 offset,
                 stat,
                 attn_span_lim,
                 attn_span_loss,
                 block_size,
                 test_only=False,
                 h_cache=None):
    X = data[:, offset: offset + block_size].contiguous()
    Y = data[:, offset + 1: offset + block_size + 1].contiguous()

    out, h_cache = model(X, h_cache)
    out = out.view(-1, out.size(-1))
    loss = torch.nn.functional.nll_loss(out, Y.view(-1))
    stat['loss'] = stat.get('loss', 0) + loss.item()

    if not test_only:
        if attn_span_loss > 0:
            loss += sum(layer.attn.attn.adaptive_span.compute_extra_loss()
                        for layer in model.module.layers)

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if attn_span_loss > 0:
            for layer in model.module.layers:
                layer.attn.attn.adaptive_span.mask.clamp_param()

    return h_cache


def _train_single_iteration(model,
                            optimizer,
                            scheduler,
                            data,
                            nb_batches,
                            full_test,
                            attn_span_lim,
                            attn_span_loss,
                            block_size,
                            test_only=False,
                            train_pos=-1,
                            h_cache=None):
    stat = dict()

    if test_only:
        model.eval()
    else:
        model.train()

    nb_batches_max = nb_batches
    if test_only:
        if full_test:
            nb_batches_max = math.ceil(data.size(1) / block_size)
        else:
            # test on fewer batches for speed-up
            nb_batches_max = max(1, nb_batches // 10)
            nb_batches_max = min(nb_batches_max,
                                 math.ceil(data.size(1) / block_size))

    actual_nb_batches = 0
    for _ in range(nb_batches_max):
        actual_nb_batches += 1
        h_cache = _train_batch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=data,
            offset=train_pos,
            stat=stat,
            attn_span_lim=attn_span_lim,
            attn_span_loss=attn_span_loss,
            block_size=block_size,
            test_only=test_only,
            h_cache=h_cache)
        train_pos += block_size
        if train_pos >= data.size(1) - block_size:
            # data position reached the end
            if full_test:
                # only test once
                break
            # randomize offset to reduce overfitting
            train_pos = random.randrange(block_size)
            # reset the cache
            for h in h_cache:
                h.fill_(0)

    for k, v in stat.items():
        stat[k] = v / actual_nb_batches
    return stat, train_pos, h_cache


def _train(device,
           model,
           optimizer,
           scheduler,
           train_data,
           val_data,
           test_data,
           checkpoint_path,
           checkpoint_freq,
           full_test,
           distributed,
           world_size,
           hidden_size,
           block_size,
           batch_size,
           nb_batches,
           nb_iter,
           attn_span_lim,
           attn_span_loss,
           plot_enabled,
           plot_env,
           plot_host):
    # create logger and plotter
    logger = Logger()
    plotter = Plotter(
        plot_enabled=plot_enabled, plot_env=plot_env, plot_host=plot_host)

    # resume training from last checkpoint
    iter_init = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        distributed=distributed)

    # hid == cache init
    # pos: 0 --> sequential /  -1 --> random
    pos = [0] * 3
    hid = [
        [
            torch.zeros(
                batch_size,
                layer.attn.attn.adaptive_span.get_cache_size(),
                hidden_size).to(device, dtype=torch.float32)
            for layer in model.module.layers
        ]
        for _ in range(3)
    ]

    if full_test:
        with torch.no_grad():
            stat_val, pos[1], hid[1] = _train_single_iteration(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data=val_data,
                nb_batches=nb_batches,
                full_test=full_test,
                attn_span_lim=attn_span_lim,
                attn_span_loss=attn_span_loss,
                block_size=block_size,
                test_only=True,
                train_pos=pos[1],
                h_cache=hid[1])
            # TODO: replace print by logger
            print('val: {:.3f}bpc'.format(stat_val['loss'] / math.log(2)))

            stat_test, pos[2], hid[2] = _train_single_iteration(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data=test_data,
                nb_batches=nb_batches,
                full_test=full_test,
                attn_span_lim=attn_span_lim,
                attn_span_loss=attn_span_loss,
                block_size=block_size,
                test_only=True,
                train_pos=pos[2],
                h_cache=hid[2])
            # TODO: replace print by logger
            print('test: {:.3f}bpc'.format(stat_test['loss'] / math.log(2)))
        return

    for iter_no in range(iter_init, nb_iter):
        t_sta = time.time()
        # here the loss includes auxilary losses such as multi-position
        # training
        stat_train, pos[0], hid[0] = _train_single_iteration(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            data=train_data,
            nb_batches=nb_batches,
            full_test=full_test,
            attn_span_lim=attn_span_lim,
            attn_span_loss=attn_span_loss,
            block_size=block_size,
            test_only=True,
            train_pos=pos[0],
            h_cache=hid[0])
        elapsed = 1000 * (time.time() - t_sta) / nb_batches
        with torch.no_grad():
            stat_val, pos[1], hid[1] = _train_single_iteration(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                data=val_data,
                nb_batches=nb_batches,
                full_test=full_test,
                attn_span_lim=attn_span_lim,
                attn_span_loss=attn_span_loss,
                block_size=block_size,
                test_only=True,
                train_pos=pos[1],
                h_cache=hid[1])

        if distributed:
            X = torch.zeros(2).to(device)
            X[0] = stat_train['loss']
            X[1] = stat_val['loss']
            torch.distributed.reduce(X, 0)
            if rank == 0:
                stat_train['loss'] = X[0] / world_size
                stat_val['loss'] = X[1] / world_size
            # why is there a continue? no plot? no checkpoint?
            else:
                continue

        span_latest = _log_iter(logger=logger,
                                iter_no=iter_no,
                                nb_batches=nb_batches,
                                stat_train=stat_train,
                                stat_val=stat_val,
                                elapsed=elapsed,
                                attn_span_loss=attn_span_loss,
                                model=model)
        _plot_iter(logger=logger, span_latest=span_latest, plotter=plotter)
        _save_iter(checkpoint_freq=checkpoint_freq,
                   checkpoint_path=actual_checkpoint_path,
                   iter_no=iter_no,
                   model=model,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   logger=logger)


def train(trainer_params,
          env_params,
          model_params,
          attn_span_params,
          optim_params,
          plotter_params,
          device,
          model,
          optimizer,
          scheduler,
          train_data,
          val_data,
          test_data):
    _train(device=device,
           model=model,
           optimizer=optimizer,
           scheduler=scheduler,
           train_data=train_data,
           val_data=val_data,
           test_data=test_data,
           checkpoint_path=trainer_params['checkpoint_path'],
           checkpoint_freq=trainer_params['checkpoint_freq'],
           full_test=trainer_params['full_test'],
           distributed=env_params['distributed'],
           world_size=env_params['world_size'],
           hidden_size=model_params['hidden_size'],
           block_size=model_params['block_size'],
           batch_size=optim_params['batch_size'],
           nb_batches=optim_params['nb_batches'],
           nb_iter=optim_params['nb_iter'],
           attn_span_lim=attn_span_params['attn_span_lim'],
           attn_span_loss=attn_span_params['attn_span_loss'],
           plot_enabled=plotter_params['plot_enabled'],
           plot_env=plotter_params['plot_env'],
           plot_host=plotter_params['plot_host'])
