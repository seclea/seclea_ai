# import os
# import unittest
# from pathlib import Path
# from unittest import mock
# from unittest.mock import mock_open, patch
#
# import responses
# from  import RequestWrapper
#
# from seclea_ai.authentication import AuthenticationService
# from seclea_ai.exceptions import AuthenticationError
#
#
# class TestAuthenticationService(unittest.TestCase):
#     def setUp(self) -> None:
#         pass
#
#     @responses.activate
#     def test_refresh_token(self):
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8000/api/token/refresh/",
#             json={"access": "dummy_access_token"},
#             status=200,
#         )
#
#         auth_service = AuthenticationService(
#             transmission=RequestWrapper(server_root_url="http://localhost:8000")
#         )
#         with patch(
#             "builtins.open",
#             new=mock_open(read_data='{"refresh": "dummy_refresh_token"}\n'),
#         ) as mock_file:
#             creds = auth_service._refresh_token()
#         mock_file.assert_called()
#         self.assertEqual(
#             creds, {"Authorization": "Bearer dummy_access_token"}, msg="Auth doesn't match"
#         )
#
#     @responses.activate
#     def test_refresh_token_expired_token(self):
#         with self.assertRaises(AuthenticationError):
#             responses.add(
#                 method=responses.POST,
#                 url="http://localhost:8000/api/token/refresh/",
#                 body="Not Authorised",
#                 status=403,
#             )
#
#             auth_service = AuthenticationService(
#                 transmission=RequestWrapper(server_root_url="http://localhost:8000")
#             )
#             with patch(
#                 "builtins.open",
#                 new=mock_open(read_data='{"refresh": "dummy_refresh_token"}\n'),
#             ):
#                 auth_service._refresh_token()
#
#     @responses.activate
#     @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
#     @mock.patch("builtins.input", autospec=True, return_value="test_user")
#     def test_login(self, mock_input, mock_getpass):
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8000/api/token/obtain/",
#             json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
#             status=200,
#         )
#
#         auth_service = AuthenticationService(
#             transmission=RequestWrapper(server_root_url="http://localhost:8000")
#         )
#
#         with patch(
#             "builtins.open",
#             new=mock_open(read_data='{"refresh": "dummy_refresh_token"}\n'),
#         ) as mock_file:
#             creds = auth_service.login()
#         self.assertEqual(
#             creds, {"Authorization": "Bearer dummy_access_token"}, msg="Auth doesn't match"
#         )
#         mock_input.assert_called_once()
#         mock_getpass.assert_called_once()
#         mock_file.assert_called_with(os.path.join(Path.home(), ".seclea/config"), "w+")
#
#     @responses.activate
#     @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
#     @mock.patch("builtins.input", autospec=True, return_value="test_user")
#     @mock.patch("seclea_ai.authentication.os.path")
#     @mock.patch("seclea_ai.authentication.os")
#     def test_handle_auth_refresh_expire(self, mock_os, mock_path, mock_input, mock_getpass):
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8000/api/token/obtain/",
#             json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
#             status=200,
#         )
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8000/api/token/refresh/",
#             body="Not Authorised",
#             status=403,
#         )
#         mock_path.isfile.return_value = False
#         mock_os.mkdir.return_value = None
#
#         auth_service = AuthenticationService(
#             transmission=RequestWrapper(server_root_url="http://localhost:8000")
#         )
#
#         with patch(
#             "builtins.open",
#             new=mock_open(
#                 read_data='{"refresh": "dummy_refresh_token", "username": "test_user"}\n'
#             ),
#         ) as mock_file:
#             creds = auth_service.handle_auth()
#         self.assertEqual(
#             creds, {"Authorization": "Bearer dummy_access_token"}, msg="Auth doesn't match"
#         )
#         mock_path.isfile.assert_called_once()
#         mock_os.mkdir.assert_called()
#         mock_input.assert_called_once()
#         mock_getpass.assert_called_once()
#         mock_file.assert_called()
