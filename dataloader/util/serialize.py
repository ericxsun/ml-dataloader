#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import msgpack
import msgpack_numpy

msgpack_numpy.patch()
assert msgpack.version >= (0, 5, 2)

__all__ = ['serialize', 'deserialize']


MAX_MSGPACK_LEN = 1000000000


class MsgpackSerializer:
    @staticmethod
    def serialize(obj):
        return msgpack.dumps(obj, use_bin_type=True)

    @staticmethod
    def deserialize(buf):
        return msgpack.loads(
            buf,
            raw=False,
            max_bin_len=MAX_MSGPACK_LEN,
            max_array_len=MAX_MSGPACK_LEN,
            max_map_len=MAX_MSGPACK_LEN,
            max_str_len=MAX_MSGPACK_LEN
        )


# Define the default serializer to be used that dumps data to bytes
serialize = MsgpackSerializer.serialize
deserialize = MsgpackSerializer.deserialize

# Define the default serializer to be used for passing data among a pair of peers.
# In this case the deserialization is known to happen only once
serialize_once = MsgpackSerializer.serialize
deserialize_once = MsgpackSerializer.deserialize
