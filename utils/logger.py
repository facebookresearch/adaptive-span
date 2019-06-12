import math
import numpy as np
import torch

class Logger(object):
    def __init__(self, args):
        self.args = args
        self.logs = dict()
        if args.plot:
            import visdom
            self.vis = visdom.Visdom(env=args.plot_env, server=args.plot_host)

    def set_state(self, state):
        self.logs = state

    def get_state(self):
        return self.logs

    def log(self, title, value, subtitle=None, opts=None):
        if subtitle is None:
            if title not in self.logs:
                self.logs[title] = {'data': [], 'type': 'line'}
            self.logs[title]['data'].append(value)
        else:
            if title not in self.logs:
                self.logs[title] = {'data': {}, 'type': 'line'}
            if subtitle not in self.logs[title]['data']:
                self.logs[title]['data'][subtitle] = []
            self.logs[title]['data'][subtitle].append(value)
        if opts is not None:
            self.logs[title]['opts'] = opts

    def update_plot(self):
        for title, v in self.logs.items():
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
                    self.vis.line(Y=data, X=self.logs['X']['data'], win=title, opts=opts)
                else:
                    self.vis.line(Y=v['data'], X=self.logs['X']['data'], win=title, opts=opts)
        self.vis.save([self.args.plot_env])

    def step(self, args, stat_train, stat_val, elapsed):
        print('{}\ttrain: {:.2f}bpc\tval: {:.2f}bpc\tms/batch: {:.1f}'.format(
            (args.ep+1)*args.nbatches,
            stat_train['loss']/math.log(2),
            stat_val['loss']/math.log(2),
            elapsed))
        self.log('train_bpc', stat_train['loss']/math.log(2))
        self.log('val_bpc', stat_val['loss']/math.log(2))
        self.log('X', (args.ep+1)*args.nbatches)

        if args.plot:
            self.update_plot()
