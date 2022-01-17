import os
import unittest
from unittest import TestCase, mock

from seclea_utils.core.transmission import RequestWrapper

SERVER_ROOT = "http://mock_server.com"
PATH_EXISTING_FILE = "/existing_file"
PATH_MISSING_FILE = "/missing_file"
SAMPLE_FILE_CONTENT = b"some content"


def mocked_server_get_file(*args, **kwargs):
    class MockResponse:
        def __init__(self, content=SAMPLE_FILE_CONTENT, status_code=200, reason=None, text=None):
            self.status_code = status_code
            self.content = content
            self.reason = reason
            self.text = text

    if args[0] == f"{SERVER_ROOT}{PATH_EXISTING_FILE}":
        return MockResponse()
    if args[0] == f"{SERVER_ROOT}{PATH_MISSING_FILE}":
        return MockResponse(status_code=400, reason="File missing", text="HTML")


class TestTransmission(unittest.TestCase):
    def setUp(self) -> None:
        pass


class TestRequestWrapper(TestCase):
    def setUp(self) -> None:
        self.data_save_path = "./tmp"

        self.request_wrapper = RequestWrapper(SERVER_ROOT)

    def tearDown(self) -> None:
        try:
            os.remove(self.data_save_path)
        except FileNotFoundError:
            pass

    @mock.patch("requests.get", side_effect=mocked_server_get_file)
    def test_load_existing_file(self, mock_get):
        self.request_wrapper.load_file(f"{PATH_EXISTING_FILE}", self.data_save_path)
        self.assertEqual(
            open(self.data_save_path, "rb").read(),
            SAMPLE_FILE_CONTENT,
            msg="File not downloaded and saved to specified path",
        )

    @mock.patch("requests.get", side_effect=mocked_server_get_file)
    def test_load_missing_file(self, mock_get):
        self.assertRaises(
            Exception, self.request_wrapper.load_file, f"{PATH_MISSING_FILE}", self.data_save_path
        )


if __name__ == "__main__":
    unittest.main()
