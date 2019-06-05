#!/usr/bin/env python3

import math


# TODO: there should be a TrainHistory/Logger object different from the plotter
# which in turn should just be a method

class Logger:
    def __init__(self):
        self._state_dict = dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def log(self, title, value, message=None):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)
        if message:
            print(message)


class Plotter:
    def __init__(self, plot_enabled, plot_env, plot_host, *args, **kwargs):
        self.plot_enabled = plot_enabled
        self.plot_env = plot_env
        if plot_enabled:
            import visdom
            self.vis = visdom.Visdom(
                env=plot_env, server=plot_host)


    def plot(self, title, X=None):
        ...


        self._state_dict = dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def _update_state_dict(self, title, value):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)

    def _update_plot(self):
        for title, data in self._state_dict.items():
            if title != 'X':
                self.vis.line(
                    X=self._state_dict['X'],
                    Y=data,
                    win=title,
                    opts={'title': title})
        self.vis.save([self.plot_env])

    def step(self, iter_no, nb_batches, stat_train, stat_val, elapsed):
        X = (iter_no + 1) * nb_batches
        # TODO: why log(2)
        train_mp_bpc = stat_train['loss'] / math.log(2)
        val_bpc = stat_val['loss'] / math.log(2)
        print('{}\ttrain: {:.2f}bpc\tval: {:.2f}bpc\tms/batch: {:.1f}'.format(
            X, train_mp_bpc, val_bpc, elapsed))
        self._update_state_dict(
            title='X',
            value=X)
        self._update_state_dict(
            title='train_mp_bpc',
            value=train_mp_bpc)
        self._update_state_dict(
            title='val_bpc',
            value=val_bpc)

        if self.plot_enabled:
            self._update_plot()


def get_plotter(plot_params):
    return Plotter(**plot_params)
