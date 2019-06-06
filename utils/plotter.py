#!/usr/bin/env python3


class Plotter:
    def __init__(self, plot_enabled, plot_env, plot_host, *args, **kwargs):
        self.plot_enabled = plot_enabled
        self.plot_env = plot_env
        if plot_enabled:
            import visdom
            self.vis = visdom.Visdom(
                env=plot_env, server=plot_host)

    def plot(self, title, Y, X=None):
        if self.plot_enabled:
            self.vis.line(X=X, Y=Y, win=title, opts={'title': title})

    def save(self, save):
        if self.plot_enabled:
            self.vis.save([self.plot_env])
