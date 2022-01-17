import io
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Union
from typing.io import BinaryIO

from seclea_utils.core import Compression


class DataManager(ABC):
    @abstractmethod
    def save_object(self, obj, reference: Any) -> Any:
        pass

    @abstractmethod
    def load_object(self, reference: Any) -> Any:
        pass


class FileManager(DataManager):
    def __init__(self, file_root: str = None):
        self.file_root = file_root

    def save_object(self, obj, reference: Any) -> Any:
        if self.file_root is not None:
            save_path = os.path.join(self.file_root, reference)
        else:
            save_path = reference
        with open(save_path, "wb") as f:
            f.write(obj)
        return save_path

    def load_object(self, reference: Any) -> Any:
        if self.file_root is not None:
            path = os.path.join(self.file_root, reference)
        else:
            path = reference
        with open(path, "rb") as f:
            return f.read()


class CompressedFileManager(DataManager):
    def __init__(self, compression: Compression, file_root: str = None):
        self.file_root = file_root
        self._compression = compression

    def save_object(self, obj: Union[bytes, BinaryIO], reference):
        if isinstance(obj, bytes):
            obj = io.BytesIO(obj)
        if self.file_root is not None:
            save_path = f"{os.path.join(self.file_root, reference)}{self._compression.extension}"
        else:
            save_path = f"{reference}{self._compression.extension}"
        with open(save_path, "wb") as out_file:
            self._compression.compress(read_stream=obj, write_stream=out_file)
        return save_path

    def load_object(self, reference) -> Any:
        if self.file_root is not None:
            path = f"{os.path.join(self.file_root, reference)}{self._compression.extension}"
        else:
            path = f"{reference}{self._compression.extension}"
        temp = tempfile.NamedTemporaryFile()
        with open(path, "rb") as in_file:
            self._compression.decompress(read_stream=in_file, write_stream=temp)
        return open(temp.name, "rb").read()
