# -*- coding: utf-8 -*-

# Controller name for the import
__name__ = 'Auth'

__doc__ = f""" introduction

    There are times you need to scrape/crawl some field on a page but the
    page requires authentication (logging in). Unless the site is using
    Basic Auth, where you can have the username and password in the url
    like http://username:1234paSSwoRd@target.site/ then you'll need to
    curl with more sophistication. Besides curl, there are other web
    tools which you can use on the command line such as links/elinks
    (elinks is an enhanced version of links which also supports
    JavaScript to a very limited extent). Links and curl will not execute
    JavaScript though, so if that's necessary to get any fields then you
    should try Selenium or CasperJS/PhantomJS instead.

    But for submitting forms and sending saved cookies, curl will suffice.

    First, you should load the page with the form you want to submit in
    Firefox. Copy the url and try loading it again, and then make sure you
    can load that url in an incognito window (to ensure you can get there
    without having already logged in). Now you can use Firefox's developer
    tools to inspect the form element: note the form submit url, and the
    fields.

    The form may have some hidden field with a value which looks random.
    This would be a type of CSRF (Cross-site request forgery) token,
    meant to protect forms from being submitted except when the form is
    generated and shown to the user. If your login form has a CSRF token
    field, then you will need to have a curl request to first load the
    form page and save the CSRF field's value and then use it to submit
    the login form (along with your username and password). The CSRF
    token may be in the form element, but it could also be in a sent
    cookie. Either way, you'll have to save some output from the initial
    request and look at the format.

    This means you not only need to retrieve the page using curl, you
    need to be able to parse the resulting html to find the csrf token
    and get its unique value. Since you are only looking for one field
    and it should be on one line, you can probably do this using common
    Unix tools like "grep", "sed", "awk", "cut". This will depend on the
    format of the html or the name of the cookie. For cookies, you can
    just send all the cookies you received instead of parsing them.

    Use curl's -d 'arg1=val1&arg2=val2&argN=valN argument to pass each
    field's value and POST it to the provided target url. For example, if
    the CSRF field was really called 'csrf' then you might POST to the
    login form like so:

    `curl -d 'username=myname&password=sEcReTwOrD123&csrf=123456' \

        http://example.com/login`
"""


#
# https://docs.python.org/3.10/library/base64.html
#
# The following documentation is automatically generated from the Python
# source files.  It may be incomplete, incorrect or include features that
# are considered implementation detail and may vary between Python
# implementations.  When in doubt, consult the module reference at the
# location listed above.
#
from base64 import encode
from encodings import utf_8
from getpass import getpass

from requests import Response

from src.core.exceptions import AuthenticationError
from seclea_utils.core import Transmission

from src.core.storage import Storage


try:
    import google.colab  # noqa F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def handle_response(res: Response, msg):
    if not res.ok:
        print(f"{msg}: {res.status_code} - {res.reason} - {res.text}")


class AuthenticationService:
    def __init__(self, transmission: Transmission):

        self._transmission = transmission

        self._db = Storage(db_name="auth_service",
                           root="." if IN_COLAB else None)

        self._path_token_obtain = "/api/token/obtain/"

        self._path_token_refresh = "/api/token/refresh/"

        self._path_token_verify = "/api/token/verify/"

        self._key_token_access = "access_token"

        self._key_token_refresh = "refresh_token"

    def authenticate(self, transmission: Transmission = None,
                     username=None, password=None):
        """ TODO: Description of the function


            Attempts to authenticate with server and then passes credential to
            specified transmission



        @param transmission:  transmission service we wish to authenticate
        @type  transmission:  class

        @return:  Description
        @rtype :  Type

        @raise e:  Description
        """

        if not self.refresh_token():
            self._obtain_initial_tokens(username=username, password=password)
        if not self.verify_token():
            raise AuthenticationError("Failed to verify token")
        transmission.cookies = self._transmission.cookies

    def verify_token(self) -> bool:
        """
        Verifies if access token in database is valid
        :return: bool valid
        """
        self._transmission.cookies = {
            self._key_token_access: self._db.get(self._key_token_access)}

        response = self._transmission.send_json(
            url_path=self._path_token_verify, obj={})
        return response.status_code == 200

    def refresh_token(self) -> bool:
        """
        Refreshes the access token by posting the refresh token.
        :return: bool success
        """
        if not self._db.get(self._key_token_refresh):
            return False
        self._transmission.cookies = {
            self._key_token_refresh: self._db.get(self._key_token_refresh)
        }
        response = self._transmission.send_json(
            url_path=self._path_token_refresh, obj={})
        self._save_response_tokens(response)
        return response.status_code == 200

    def _request_user_credentials(self) -> dict:
        """
        Gets user credentials manually
        :return:
        """
        return {"username": input("Username: "),
                "password": getpass("Password: ")}

    def _save_response_tokens(self, response) -> None:
        """
        Saves refresh and access tokens into db
        :param response:
        :return:
        """
        if self._key_token_refresh in response.cookies:
            self._db.write(self._key_token_refresh,
                           response.cookies[self._key_token_refresh])
        if self._key_token_access in response.cookies:
            self._db.write(self._key_token_access,
                           response.cookies[self._key_token_access])

    def _obtain_initial_tokens(self, username=None, password=None):
        if username is None or password is None:
            credentials = self._request_user_credentials()
        else:
            credentials = {"username": username, "password": password}
        response = self._transmission.send_json(
            url_path=self._path_token_obtain, obj=credentials)
        if response.status_code != 200:
            raise AuthenticationError(
                f"status:{response.status_code}, content:{response.content}")
        self._save_response_tokens(response)
