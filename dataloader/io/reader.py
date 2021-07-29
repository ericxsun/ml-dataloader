#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import abc
from typing import Sequence
from typing import Union


class Reader:
    """Abstract class to define interface APIs to load files"""

    @abc.abstractmethod
    def read(self, data: Union[Sequence[str], str], **kwargs):
        """Read data from specified file or files

        Args:
            data: file name or a list of file names to be read
            **kwargs: additional args for actual `read` API of 3rd party libs

        Returns:
            numpy array of data
        """
        raise NotImplementedError(f'Subclass {self.__class__.__name__} must implement this method')
