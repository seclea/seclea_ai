import logging
from getpass import getpass

from requests import Session

from .exceptions import AuthenticationError
from ..lib.seclea_utils.core.transmission import Transmission
from .storage import Storage

try:
    import google.colab  # noqa F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


logger = logging.getLogger(__name__)


class AuthenticationService:
    def __init__(self, url: str, session: Session):
        self._url = url
        self._session = session
        self._db = Storage(db_name="auth_service", root="." if IN_COLAB else None)
        self._path_token_obtain = (
            "/api/token/obtain/"  # nosec - bandit thinks this is a pw or key..
        )
        self._path_token_refresh = (
            "/api/token/refresh/"  # nosec - bandit thinks this is a pw or key..
        )
        self._path_token_verify = (
            "/api/token/verify/"  # nosec - bandit thinks this is a pw or key..
        )
        self._key_token_access = "access_token"  # nosec - bandit thinks this is a pw or key..
        self._key_token_refresh = "refresh_token"  # nosec - bandit thinks this is a pw or key..

    def authenticate(self, transmission: Transmission = None, username=None, password=None):
        """
        Attempts to authenticate with server and then passes credential to specified transmission

        :param transmission: transmission service we wish to authenticate
        :return:
        """
        if not self.verify_token():
            if not self.refresh_token():
                self._obtain_initial_tokens(username=username, password=password)

    def verify_token(self) -> bool:
        """
        Verifies if access token in database is valid
        :return: bool valid
        """
        cookies = {self._key_token_access: self._db.get(self._key_token_access)}
        self._session.cookies.update(cookies)
        logger.debug(f"Cookies: {cookies}")
        response = self._session.post(url=f"{self._url}/{self._path_token_verify}")
        return response.status_code == 200

    def refresh_token(self) -> bool:
        """
        Refreshes the access token by posting the refresh token.
        :return: bool success
        """
        if not self._db.get(self._key_token_refresh):
            return False
        cookies = {self._key_token_refresh: self._db.get(self._key_token_refresh)}
        self._session.cookies.update(cookies)
        response = self._session.post(url=f"{self._url}/{self._path_token_refresh}")
        self._save_response_tokens(response)
        return response.status_code == 200

    @staticmethod
    def _request_user_credentials() -> dict:
        """
        Gets user credentials manually
        :return:
        """
        return {"username": input("Username: "), "password": getpass("Password: ")}

    def _save_response_tokens(self, response) -> None:
        """
        Saves refresh and access tokens into db
        :param response:
        :return:
        """
        cookies = response.cookies.get_dict()
        if self._key_token_refresh in cookies:
            self._db.write(self._key_token_refresh, cookies[self._key_token_refresh])
        if self._key_token_access in cookies:
            self._db.write(self._key_token_access, cookies[self._key_token_access])

    def _obtain_initial_tokens(self, username=None, password=None):
        if username is None or password is None:
            credentials = self._request_user_credentials()
        else:
            logger.warning("Warning - Avoid storing credentials in code where possible!")
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
