#!/usr/bin/env python3


class Logger:
    def __init__(self):
        self._state_dict = dict()

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict

    def log(self, title, value):
        if title not in self._state_dict:
            self._state_dict[title] = []
        self._state_dict[title].append(value)

    def __contains__(self, title):
        return title in self._state_dict

    def get_data(self, title):
        if title not in self:
            raise KeyError(title)
        return self._state_dict[title]
