import os
import unittest

from seclea_ai.internal.config import human_2_numeric_bytes, read_yaml
from seclea_ai.lib.seclea_utils.core.file_management import CustomNamedTemporaryFile

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_path = os.path.join(base_dir, "")
print(folder_path)


class TestSecleaAIThreading(unittest.TestCase):
    def test_human_2_numeric_bytes_success(self):
        # ARRANGE
        test_values = ["100k", "20m", "30Gb", "4tB", "55PB"]
        expected_results = [int(100e3), int(20e6), int(30e9), int(4e12), int(55e15)]

        # ACT
        results = list()
        for value in test_values:
            results.append(human_2_numeric_bytes(value))

        # ASSERT
        for result, expected_result in zip(results, expected_results):
            self.assertEqual(result, expected_result)

    def test_human_2_numeric_bytes_error(self):
        # ASSERT
        with self.assertRaises(ValueError):
            # ARRANGE
            test_values = ["100"]  # list to allow testing more error cases if needed.

            # ACT
            results = list()
            for value in test_values:
                results.append(human_2_numeric_bytes(value))

    def test_read_yaml(self):
        # ARRANGE
        expected_result = {"test_value": "100mest"}
        with CustomNamedTemporaryFile() as temp:
            with open(temp.name, "w") as write_temp:
                write_temp.write("test_value: 100mest")

            # ACT
            document = read_yaml(temp.name)

        # ASSERT
        self.assertEqual(document, expected_result)

    def test_read_yaml_empty(self):
        # ARRANGE
        expected_result = None
        with CustomNamedTemporaryFile() as temp:
            with open(temp.name, "w") as write_temp:
                write_temp.write("")

            # ACT
            document = read_yaml(temp.name)

        # ASSERT
        self.assertEqual(document, expected_result)

    def test_read_yaml_no_file(self):
        # ASSERT
        with self.assertRaises(FileNotFoundError):
            # ARRANGE
            # set up temp file to get file name that won't exist
            with CustomNamedTemporaryFile() as temp:
                pass

            # ACT
            # try and read non-existent file
            read_yaml(temp.name)


if __name__ == "__main__":
    unittest.main()
