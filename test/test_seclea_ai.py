# import os
# import unittest
#
# from unittest import mock
# from unittest.mock import mock_open, patch
#
# import pandas as pd
# import responses
#
# from seclea_ai import SecleaAI
#
#
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# folder_path = os.path.join(base_dir, "")
# print(folder_path)
#
#
# class TestSecleaAI(unittest.TestCase):
#
#     def test_check_features_different_names(self):
#         parent_metadata = {
#             "features": ["months_as_customer", "age", "policy_number", "policy_bind_date"]
#         }
#         test_dataset = pd.DataFrame([[3, 34, 339203, 34], [7, 32, 339103, 3]], columns=[1, 2, 3, 4])
#         metadata = {}
#         metadata = SecleaAI._check_features(dataset=test_dataset, metadata=metadata, parent_metadata=parent_metadata)
#         print(metadata)
#
#     def test_check_features_different_names_different_len(self):
#         parent_metadata = {
#             "features": ["months_as_customer", "age", "policy_number", "policy_bind_date"]
#         }
#         test_dataset = pd.DataFrame([[3, 34, 339203, 34], [7, 32, 339103, 3]], columns=[1, 2, 3, 4])
#         metadata = {}
#         metadata = SecleaAI._check_features(dataset=test_dataset, metadata=metadata, parent_metadata=parent_metadata)
#         print(metadata)
#

#     @responses.activate
#     @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
#     @mock.patch("builtins.input", autospec=True, return_value="test_user")
#     def test_init_seclea_object(self, mock_input, mock_getpass) -> None:
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8010/api/token/refresh/",
#             json={"access": "dummy_access_token"},
#             status=200,
#         )
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8010/api/token/obtain/",
#             json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
#             status=200,
#         )
#         responses.add(
#             method=responses.GET,
#             url="http://localhost:8000/collection/projects?name=test-project",
#             json=[{"id": 1, "name": "test-project"}],
#             status=200,
#         )
#         responses.add(
#             method=responses.GET,
#             url="http://localhost:8000/collection/models",
#             json=[{"id": 1, "name": "GBM-1"}],
#             status=200,
#         )
#         responses.add(
#             method=responses.GET,
#             url="http://localhost:8000/collection/datasets",
#             json=[{"id": 1, "name": "Fraud detection"}],
#             status=200,
#         )
#         with patch(
#             "builtins.open",
#             new=mock_open(
#                 read_data='{"refresh": "dummy_refresh_token", "username": "test_user"}\n'
#             ),
#         ) as mock_file:
#             SecleaAI(
#                 project_name="test-project",
#                 organization="Onespan",
#                 platform_url="http://localhost:8000",
#                 auth_url="http://localhost:8010",
#             )
#         mock_file.assert_called()
#
#     @responses.activate
#     @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
#     @mock.patch("builtins.input", autospec=True, return_value="test_user")
#     def test_getting_project_fail(self, mock_input, mock_getpass) -> None:
#         """Test using an Project name that does not exist and sending new project fails raises a ValueError"""
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8010/api/token/refresh/",
#             json={"access": "dummy_access_token"},
#             status=200,
#         )
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8010/api/token/obtain/",
#             json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
#             status=200,
#         )
#         responses.add(
#             method=responses.GET,
#             url="http://localhost:8000/collection/projects?name=New Project",
#             status=400,
#         )
#         with self.assertRaises(ValueError):
#             SecleaAI(
#                 project_name="New Project",
#                 organization="Onespan",
#                 platform_url="http://localhost:8000",
#                 auth_url="http://localhost:8010",
#             )
#
#     @responses.activate
#     @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
#     @mock.patch("builtins.input", autospec=True, return_value="test_user")
#     def test_upload_project_fail(self, mock_input, mock_getpass) -> None:
#         """Test using an Project name that does not exist and sending new project fails raises a ValueError"""
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8010/api/token/refresh/",
#             json={"access": "dummy_access_token"},
#             status=200,
#         )
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8010/api/token/obtain/",
#             json={"access": "dummy_access_token", "refresh": "dummy_refresh_token"},
#             status=200,
#         )
#         responses.add(
#             method=responses.GET,
#             url="http://localhost:8000/collection/projects?name=New Project",
#             status=200,
#             json=[],
#         )
#         responses.add(
#             method=responses.GET,
#             url="http://localhost:8000/collection/models",
#             json=[{"id": 1, "name": "GBM-1"}],
#             status=200,
#         )
#         responses.add(
#             method=responses.GET,
#             url="http://localhost:8000/collection/datasets",
#             json=[{"id": 1, "name": "Fraud detection"}],
#             status=200,
#         )
#         responses.add(
#             method=responses.POST,
#             url="http://localhost:8000/collection/projects",
#             status=403,
#         )
#         with self.assertRaises(ValueError):
#             SecleaAI(
#                 project_name="New Project",
#                 organization="Onespan",
#                 platform_url="http://localhost:8000",
#                 auth_url="http://localhost:8010",
#             )
#
#
#
# if __name__ == "__main__":
#     unittest.main()
