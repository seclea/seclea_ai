import logging
import traceback
from getpass import getpass
from pathlib import Path

from peewee import SqliteDatabase
from requests import Response, Session

from ..internal.exceptions import AuthenticationError
from ..internal.persistence.auth_credentials import AuthCredentials

try:
    import google.colab  # noqa F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


logger = logging.getLogger(__name__)


def handle_response(res: Response, msg):
    if not res.ok:
        print(f"{msg}: {res.status_code} - {res.reason} - {res.text}")


# TODO fix this - the flow either here or in the threads is not working consistently.
class AuthenticationService:
    def __init__(self, url: str, session: Session):
        self._url = url
        self._db = SqliteDatabase(
            Path.home() / ".seclea" / "seclea_ai.db",
            thread_safe=True,
            pragmas={"journal_mode": "wal"},
        )
        self._session = session
        self._path_token_obtain = "api/token/obtain/"  # nosec - bandit thinks this is a pw or key..
        self._path_token_refresh = (
            "api/token/refresh/"  # nosec - bandit thinks this is a pw or key..
        )
        self._path_token_verify = "api/token/verify/"  # nosec - bandit thinks this is a pw or key..
        self._key_token_access = "access_token"  # nosec - bandit thinks this is a pw or key..
        self._key_token_refresh = "refresh_token"  # nosec - bandit thinks this is a pw or key..

    def authenticate(self, username=None, password=None):
        """
        Attempts to authenticate with server and then passes credential to specified transmission

        :param transmission: transmission service we wish to authenticate

        :return:
        """
        with self._db.atomic():
            if not self.verify_token():
                if not self.refresh_token():
                    self._obtain_initial_tokens(username=username, password=password)

    def verify_token(self) -> bool:
        """
        Verifies if access token in database is valid
        :return: bool valid
        """
        if AuthCredentials.get_or_none(AuthCredentials.key == self._key_token_access) is None:
            return False
        cookies = {
            self._key_token_access: AuthCredentials.get(
                AuthCredentials.key == self._key_token_access
            ).value
        }

        self._session.cookies.update(
            cookies
        )  # TODO check cookie first - keep original for some reason?
        logger.debug(f"Cookies: {cookies}")  # TODO remove

        try:
            response = self._session.post(url=f"{self._url}/{self._path_token_verify}")
        except Exception:
            traceback.print_exc()
            raise
            # self._session.cookies.pop(self._key_token_access)  # reset the cookie. TODO check this flow
            # return False
        return response.status_code == 200

    def refresh_token(self) -> bool:
        """
        Refreshes the access token by posting the refresh token.

        :return: bool Success
        """
        if AuthCredentials.get_or_none(AuthCredentials.key == self._key_token_refresh) is None:
            return False
        cookies = {
            self._key_token_refresh: AuthCredentials.get(
                AuthCredentials.key == self._key_token_refresh
            ).value
        }
        # TODO add more validation logic like in verify? - factor out?
        self._session.cookies.update(cookies)
        response = self._session.post(url=f"{self._url}/{self._path_token_refresh}")
        self._save_response_tokens(response)
        return response.status_code == 200

    @staticmethod
    def _request_user_credentials() -> dict:
        """
        Gets user credentials manually

        :return: dict Username and password
        """
        return {"username": input("Username: "), "password": getpass("Password: ")}

    def _save_response_tokens(self, response) -> None:
        """
        Saves refresh and access tokens into db

        :param response:

        :return: None
        """
        cookies = response.cookies.get_dict()
        if self._key_token_refresh in cookies:
            AuthCredentials.get_or_create(
                key=self._key_token_refresh,
                defaults={"value": cookies[self._key_token_refresh]},
            )
        if self._key_token_access in cookies:
            AuthCredentials.get_or_create(
                key=self._key_token_access,
                defaults={"value": cookies[self._key_token_access]},
            )

    def _obtain_initial_tokens(self, username=None, password=None) -> None:
        """
        Wrapper method to get initial tokens, either with passed in credentials. (For non secure scripting use only)
        This method saves the tokens in the db.

        :param username: str Username

        :param password: str Password

        :return: None
        """
        if username is None or password is None:
            credentials = self._request_user_credentials()
        else:
            print("Warning - Avoid storing credentials in code where possible!")
            credentials = {"username": username, "password": password}
        response = self._session.post(
            url=f"{self._url}/{self._path_token_obtain}", data=credentials
        )
        logger.debug(
            f"Initial Tokens - Status: {response.status_code} - content {response.content} - cookies - {response.cookies}"
        )
        if response.status_code != 200:
            raise AuthenticationError(f"status:{response.status_code}, content:{response.content}")
        self._save_response_tokens(response)
