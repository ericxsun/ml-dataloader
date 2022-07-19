#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from dataloader.util.misc import get_from_registry

_reader_registry = {}


def get_reader_class(name):
    return get_from_registry(name, _reader_registry)


def register_reader(name: str):
    def wrap(cls):
        if name not in _reader_registry:
            _reader_registry[name] = cls
        else:
            raise Exception(f'exist registered reader with name: `{name}`')

        return cls

    return wrap
