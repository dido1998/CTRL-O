import abc
import gzip
import io
import json
import tarfile
from typing import Any, List, Tuple

import numpy as np


class Handler(abc.ABC):
    @abc.abstractmethod
    def is_responsible(self, instance: Any) -> bool:
        pass

    @abc.abstractmethod
    def __call__(self, name: str, objs: Any) -> Tuple[str, bytes]:
        pass


class NumpyHandler(Handler):
    def is_responsible(self, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def __call__(self, name: str, obj: np.ndarray) -> bool:
        with io.BytesIO() as stream:
            np.save(stream, obj)
            return f"{name}.npy", stream.getvalue()


class JsonHandler(Handler):
    def is_responsible(self, obj: Any) -> bool:
        return isinstance(obj, (list, dict))

    def __call__(self, name: str, obj: np.ndarray) -> bool:
        return f"{name}.json", json.dumps(obj).encode("utf-8")


class GzipHandler(Handler):
    def __init__(self, compresslevel=5):
        self.compresslevel = compresslevel

    def is_responsible(self, obj: Any) -> bool:
        # Only compress if file is larger than a block, otherwise compression
        # is useless because the file will anyway use 512 bytes of space.
        return isinstance(obj, bytes) and len(obj) > tarfile.BLOCKSIZE

    def __call__(self, name: str, obj: bytes) -> bool:
        return f"{name}.gz", gzip.compress(obj, compresslevel=self.compresslevel)


DEFAULT_HANDLERS = [NumpyHandler(), JsonHandler(), GzipHandler()]


def convert_to_bytes(name, obj, handlers: List[Handler] = DEFAULT_HANDLERS):
    for handler in handlers:
        if handler.is_responsible(obj):
            name, obj = handler(name, obj)
    return name, obj
