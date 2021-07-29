#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import importlib
import types


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    refer: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
    """
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals

        super(LazyLoader, self).__init__(name)

    def _load(self):
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
