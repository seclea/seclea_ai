from getpass import getpass

from requests import Response, Session
from seclea_utils.core import Transmission

from seclea_ai.exceptions import AuthenticationError

from .storage import Storage


def handle_response(res: Response, msg):
    if not res.ok:
        print(f"{msg}: {res.status_code} - {res.reason} - {res.text}")


def singleton(cls):
    """Decorator to ensures a class follows the singleton pattern.

    Example:
        @singleton
        class MyClass:
            ...
    """
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class AuthenticationService:
    def __init__(self, url: str, session: Session):
        self._instance = None
        self._session = session
        self._db = Storage(db_name="auth_service")
        self._url = url
        self._path_token_obtain = "/api/token/obtain/"
        self._path_token_refresh = "/api/token/refresh/"
        self._path_token_verify = "/api/token/verify/"
        self._key_token_access = "access_token"
        self._key_token_refresh = "refresh_token"

    def authenticate(self, transmission: Transmission = None, username=None, password=None):
        """
        Attempts to authenticate with server and then passes credential to specified transmission

        :param transmission: transmission service we wish to authenticate

        :return:
        """
        if not self.refresh_stored_token():
            self._obtain_initial_tokens(username=username, password=password)
        if not self.verify_token():
            raise AuthenticationError("Failed to verify token")
        # transmission.cookies = self._transmission.cookies

    def verify_token(self) -> bool:
        """
        Verifies if the access token a transmission object has is valid.

        :param transmission: The transmission object containing the access token

        :return: bool True if valid or False
        """
        # if self._key_token_access not in transmission.cookies:
        #     return False
        # self._transmission.cookies = {
        #     self._key_token_access: transmission.cookies[self._key_token_access]
        # }
        response = self._session.post(url=f"{self._url}{self._path_token_verify}")
        return response.status_code == 200

    def verify_stored_token(self) -> bool:
        """
        Verifies if access token in database is valid

        :return: bool True if valid or False
        """
        if not self._db.get(self._key_token_access):
            return False
        self._transmission.cookies = {
            self._key_token_access: self._db.get(self._key_token_access, default="")
        }

        response = self._session.post(
            url=f"{self._url}{self._path_token_verify}",
            cookies={self._key_token_access: self._db.get(self._key_token_access, default="")},
        )
        return response.status_code == 200

    def refresh_token(self, transmission: Transmission) -> bool:
        """
        Refreshes the access token for the transmission if successful.

        :param transmission: Transmission The transmission which needs the refreshed token.

        :return: bool Success
        """
        if not self._db.get(self._key_token_refresh):
            return False
        # self._transmission.cookies = {
        #     self._key_token_refresh: self._db.get(self._key_token_refresh)
        # }
        response = self._session.post(
            url=f"{self.url}{self._path_token_refresh}",
            cookies={self._key_token_refresh: self._db.get(self._key_token_refresh)},
        )
        # self._save_response_tokens(response)
        # if response.status_code == 200:
        #     transmission.cookies = {self._key_token_access: self._db.get(self._key_token_access)}
        return response.status_code == 200

    def refresh_stored_token(self) -> bool:
        """
        Refreshes the stored access token by posting the refresh token.

        :return: bool Success
        """
        if not self._db.get(self._key_token_refresh):
            return False
        # self._transmission.cookies = {
        #     self._key_token_refresh: self._db.get(self._key_token_refresh)
        # }
        response = self._session.post(
            url=f"{self._url}{self._path_token_refresh}",
            cookies={self._key_token_refresh: self._db.get(self._key_token_refresh)},
        )
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
        if self._key_token_refresh in response.cookies:
            self._db.write(self._key_token_refresh, response.cookies[self._key_token_refresh])
        if self._key_token_access in response.cookies:
            self._db.write(self._key_token_access, response.cookies[self._key_token_access])

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
        response = self._session.post(url=f"{self._url}{self._path_token_obtain}", json=credentials)
        if response.status_code != 200:
            raise AuthenticationError(f"status:{response.status_code}, content:{response.content}")
        self._save_response_tokens(response)
