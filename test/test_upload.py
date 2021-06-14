import unittest

from seclea_ai.seclea_ai import SecleaAI


class TestUpload(unittest.TestCase):
    def test_create_project_no_name(self) -> None:

        with self.assertRaises(Exception) as context:
            seclea = SecleaAI(
                plat_url="https://tristar-platform.seclea.com",
                auth_url="https://tristar-auth.seclea.com",
            )
            seclea.login(username="onespanadmin", password="logmein1")  # nosec
            seclea.create_project(description="A test project")
            self.assertTrue("No project name specified, please provide one" in context.exception)


if __name__ == "__main__":
    unittest.main()
