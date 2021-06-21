import unittest
from unittest import mock
from unittest.mock import mock_open, patch

import responses

from seclea_ai import SecleaAI


class TestSecleaAI(unittest.TestCase):
    @responses.activate
    @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
    @mock.patch("builtins.input", autospec=True, return_value="test_user")
    def test_init_seclea_object(self, mock_input, mock_getpass) -> None:
        responses.add(
            method=responses.POST,
            url="http://localhost:8010/api/token/refresh/",
            json={"access": "dummy_access_token"},
            status=200,
        )
        responses.add(
            method=responses.POST,
            url="http://localhost:8010/api/token/obtain/",
            json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
            status=200,
        )
        responses.add(
            method=responses.GET,
            url="http://localhost:8000/collection/projects/test-project",
            json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
            status=200,
        )
        with patch(
            "builtins.open",
            new=mock_open(
                read_data='{"refresh": "dummy_refresh_token", "username": "test_user"}\n'
            ),
        ) as mock_file:
            SecleaAI(
                project_name="test-project",
                plat_url="http://localhost:8000",
                auth_url="http://localhost:8010",
            )
        mock_file.assert_called()


if __name__ == "__main__":
    unittest.main()
