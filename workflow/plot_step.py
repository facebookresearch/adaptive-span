#!/usr/bin/env python3

import math
import visdom
import numpy as np
import torch


class Plotter:
    def __init__(self, plot_enabled, plot_env, plot_host, *args, **kwargs):
        self.plot_enabled = plot_enabled
        self.plot_env = plot_env
        if plot_enabled:
            self.vis = visdom.Visdom(
                env=plot_env, server=plot_host)
        self.plots = dict()

    def set_state(self, state):
        self.plots = state

    def get_state(self):
        return self.plots

    def log(self, title, value, subtitle=None, opts=None):
        if subtitle is None:
            if title not in self.plots:
                self.plots[title] = {'data': [], 'type': 'line'}
            self.plots[title]['data'].append(value)
        else:
            if title not in self.plots:
                self.plots[title] = {'data': {}, 'type': 'line'}
            if subtitle not in self.plots[title]['data']:
                self.plots[title]['data'][subtitle] = []
            self.plots[title]['data'][subtitle].append(value)
        if opts is not None:
            self.plots[title]['opts'] = opts

    def hist(self, title, data, sub_ind=None):
        if sub_ind is None:
            self.plots[title] = {'data': data, 'type': 'hist'}
        else:
            if title not in self.plots:
                self.plots[title] = {'data': {}, 'type': 'hist'}
            self.plots[title]['data'][sub_ind] = data

    def image(self, title, data):
        self.plots[title] = {'data': data, 'type': 'image'}

    def update_plot(self):
        for title, v in self.plots.items():
            opts = {'title': title}
            if 'opts' in v:
                opts.update(v['opts'])
            if v['type'] == 'line':
                if title == 'X':
                    pass
                elif isinstance(v['data'], dict):
                    data = []
                    legend = []
                    for st, d in v['data'].items():
                        data.append(d)
                        legend.append(st)
                    data = np.transpose(np.array(data))
                    opts['legend'] = legend
                    self.vis.line(
                        Y=data,
                        X=self.plots['X']['data'],
                        win=title,
                        opts=opts)
                else:
                    self.vis.line(
                        Y=v['data'],
                        X=self.plots['X']['data'],
                        win=title,
                        opts=opts)
            elif v['type'] == 'image':
                opts.update({'width': 200, 'height': 200})
                self.vis.image(v['data'], win=title, opts=opts)
            elif v['type'] == 'hist':
                if isinstance(v['data'], dict):
                    data = [v['data'][i] for i in range(len(v['data']))]
                    data = torch.stack(data)
                    self.vis.histogram(data, win=title, opts=opts)
                else:
                    self.vis.histogram(v['data'], win=title, opts=opts)
        self.vis.save([self.plot_env])

    def step(self, ep, nb_batches, stat_train, stat_val, elapsed):
        print('{}\ttrain: {:.2f}bpc\tval: {:.2f}bpc\tms/batch: {:.1f}'.format(
            (ep + 1) * nb_batches,
            stat_train['loss'] / math.log(2),
            stat_val['loss'] / math.log(2),
            elapsed))
        self.log('train_mp_bpc', stat_train['loss'] / math.log(2))
        self.log('val_bpc', stat_val['loss'] / math.log(2))
        self.log('X', (ep + 1) * nb_batches)

        if self.plot_enabled:
            self.update_plot()


def get_plotter(plot_params):
    return Plotter(**plot_params)
