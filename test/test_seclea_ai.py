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
            url="http://localhost:8000/collection/projects?name=test-project",
            json=[{"id": 1, "name": "test-project"}],
            status=200,
        )
        responses.add(
            method=responses.GET,
            url="http://localhost:8000/collection/models",
            json=[{"id": 1, "name": "GBM-1"}],
            status=200,
        )
        responses.add(
            method=responses.GET,
            url="http://localhost:8000/collection/datasets",
            json=[{"id": 1, "name": "Fraud detection"}],
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
                framework="sklearn",
                plat_url="http://localhost:8000",
                auth_url="http://localhost:8010",
            )
        mock_file.assert_called()

    def test_init_seclea_object_bad_framework(self) -> None:
        """Test using an incorrect framework raises a ValueError"""
        with self.assertRaises(ValueError):
            SecleaAI(
                project_name="test-project",
                framework="something incorrect",
                plat_url="http://localhost:8000",
                auth_url="http://localhost:8010",
            )

    @responses.activate
    @mock.patch("seclea_ai.authentication.getpass", return_value="test_pass")
    @mock.patch("builtins.input", autospec=True, return_value="test_user")
    def test_init_project_fail(self, mock_input, mock_getpass) -> None:
        """Test using an incorrect framework raises a ValueError"""
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
            url="http://localhost:8000/collection/projects?name=New Project",
            status=400,
        )
        responses.add(
            method=responses.GET,
            url="http://localhost:8000/collection/models",
            json=[{"id": 1, "name": "GBM-1"}],
            status=200,
        )
        responses.add(
            method=responses.GET,
            url="http://localhost:8000/collection/datasets",
            json=[{"id": 1, "name": "Fraud detection"}],
            status=200,
        )
        responses.add(
            method=responses.POST,
            url="http://localhost:8000/collection/projects",
            status=403,
        )
        with self.assertRaises(ValueError):
            SecleaAI(
                project_name="New Project",
                framework="sklearn",
                plat_url="http://localhost:8000",
                auth_url="http://localhost:8010",
            )


if __name__ == "__main__":
    unittest.main()
