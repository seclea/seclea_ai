import unittest
from unittest import mock
from unittest.mock import mock_open, patch

import responses
from seclea_utils.data.transmission import RequestWrapper

from seclea_ai.authentication import AuthenticationService


class TestAuthenticationService(unittest.TestCase):
    def setUp(self) -> None:
        pass

    @responses.activate
    @mock.patch("seclea_ai.authentication.os.path")
    @mock.patch("seclea_ai.authentication.os", autospec=True)
    def test_refresh_token(self, mock_os, mock_os_path):
        responses.add(
            method=responses.POST,
            url="http://localhost:8000/api/token/refresh/",
            json={"access": "dummy_access_token"},
            status=200,
        )
        # test config file existing
        mock_os_path.is_file.return_value = False
        mock_os.mkdir.return_value = None

        auth_service = AuthenticationService(
            transmission=RequestWrapper(server_root_url="http://localhost:8000")
        )
        with patch(
            "builtins.open",
            new=mock_open(
                read_data='{"refresh": "dummy_refresh_token", "username": "test_user"}\n'
            ),
        ) as mock_file:
            username, creds = auth_service._refresh_token()
        mock_file.assert_called()
        self.assertEqual(username, "test_user", msg="Username not the same as that in config file")
        self.assertEqual(creds, {"Authorization": "Bearer dummy_access_token"})
