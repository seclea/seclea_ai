import unittest
from unittest import mock

from seclea_ai import SecleaAI


class TestUpload(unittest.TestCase):
    @mock.patch("seclea_ai.seclea_ai.RequestWrapper", autospec=True)
    @mock.patch("getpass.fallback_getpass", autospec=True, return_value="test_pass")
    @mock.patch("builtins.input", autospec=True, return_value="test_user")
    def test_create_project(self, mock_input, mock_getpass, mock_trans) -> None:

        SecleaAI(
            project_name="test_project",
            plat_url="http://localhost:8000",
            auth_url="http://localhost:8010",
        )
        self.assertTrue(mock_trans.get.called, "something")


if __name__ == "__main__":
    unittest.main()
