from abc import ABC, abstractmethod

from zstandard import ZstdCompressor, ZstdDecompressor

from seclea_utils.core.exceptions import CompressionError, DecompressionError
from seclea_utils.core.typing import BytesStream


class Compression(ABC):
    def __init__(self, chunk_size=256 * (10 ** 3), compression_ext=".comp"):
        self.chunk_size = chunk_size
        self.extension = compression_ext

    @abstractmethod
    def compress(self, read_stream: BytesStream, write_stream: BytesStream):
        pass

    @abstractmethod
    def decompress(self, read_stream: BytesStream, write_stream: BytesStream):
        pass


# Start of concrete implementations #
class Zstd(Compression):
    def __init__(self):
        super().__init__(compression_ext=".zstd")
        self.cctx = ZstdCompressor()
        self.dctx = ZstdDecompressor()

    def compress(self, read_stream, write_stream):
        try:
            read_stream.seek(0, 2)
            size = read_stream.tell()
            read_stream.seek(0, 0)
            self.cctx.copy_stream(read_stream, write_stream, size=size, write_size=self.chunk_size)
        except Exception:
            raise CompressionError("An error occurred during compression.")

    def decompress(self, read_stream, write_stream):
        try:
            with self.dctx.stream_reader(read_stream) as rs:
                rb = rs.read(self.chunk_size)
                while rb:
                    write_stream.write(rb)
                    rb = rs.read(self.chunk_size)
        except Exception:
            raise DecompressionError("An error occurred during decompression")


#   REFERENCE IF WE WANT TO HAVE MORE CONTROL OVER COMPRESSION IN THE FUTURE   ###
# while rb:
#     i += 1
#     print((i * self.chunk_size) / (2000 * (10 ** 6)))
#     t = time()
#     rb = read_stream.read(self.chunk_size)
#     print(rb)
#     print("read:", time() - t)
#     t = time()
#     out = self.chunker.compress(rb)
#     for o in out:
#         print(o)
#     print("compress:", time() - t)
#     t = time()
#
#     wr = self.cctx.stream_writer(write_stream, write_size=self.chunk_size)
#     print("decl:", time() - t)
#     t = time()
#     for o in out:
#         print("iter:", time() - t)
#         t = time()
#         wr.write(o)
#         print("write-chunk", time() - t)
#         t = time()
#
#     t = time()
#     print("write:", time() - t)
#
# for out in self.chunker.finish():
#     write_stream.write(out)
