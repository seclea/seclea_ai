import os.path
import unittest

from seclea_utils.core import Zstd


class TestZstd(unittest.TestCase):
    def setUp(self) -> None:
        self.comp = Zstd()
        self.original = "test/example_files/bee_movie.txt"
        self.compressed = "test/example_files/bee_movie.zstd"
        self.decompressed = "test/example_files/bee_movie_d.txt"

    def test_compress_decompress(self):
        orig_size = os.path.getsize(self.original)
        with open(self.original, "rb") as fs, open(self.compressed, "wb") as fo:
            self.comp.compress(fs, fo)
        self.assertLess(
            os.path.getsize(self.compressed), orig_size, "File post compression not smaller"
        )
        with open(self.compressed, "rb") as ch, open(self.decompressed, "wb") as co:
            self.comp.decompress(ch, co)
        self.assertEqual(
            os.path.getsize(self.decompressed),
            orig_size,
            "File post decompression not original size",
        )


if __name__ == "__main__":
    unittest.main()
