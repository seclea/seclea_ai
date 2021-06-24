import json
import os
import stat
from getpass import getpass
from pathlib import Path
from typing import Dict, Tuple

from requests import Response

from seclea_ai.exceptions import AuthenticationError

AuthenticationCredentials = Dict[str, str]


def handle_response(res: Response, msg):
    if not res.ok:
        print(f"{msg}: {res.status_code} - {res.reason} - {res.text}")


class AuthenticationService:
    def __init__(self, transmission):
        self._access = None
        self._transmission = transmission

    def handle_auth(self) -> Tuple[str, AuthenticationCredentials]:
        if os.path.isfile(os.path.join(Path.home(), ".seclea/config")):
            try:
                username, creds = self._refresh_token()
            except AuthenticationError:
                username, creds = self.login()
        else:
            try:
                os.mkdir(
                    os.path.join(Path.home(), ".seclea"),
                    stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_ISVTX,
                )  # set mode to allow user and group rw only
            except FileExistsError:
                # do nothing.
                pass
            username, creds = self.login()
        return username, creds

    def login(self) -> Tuple[str, AuthenticationCredentials]:
        username = input("Username: ")
        password = getpass("Password: ")
        credentials = {"username": username, "password": password}
        response = self._transmission.send_json(url_path="/api/token/obtain/", obj=credentials)
        try:
            response_content = json.loads(response.content.decode("utf-8"))
        except Exception as e:
            print(e)
            raise json.decoder.JSONDecodeError("INVALID CREDENTIALS: ", str(credentials), 1)
        self._access = response_content.get("access")
        if self._access is not None:
            # note from this api access and refresh are returned together. Something to be aware of though.
            # TODO refactor when adding more to config
            with open(os.path.join(Path.home(), ".seclea/config"), "w+") as f:
                f.write(
                    json.dumps({"refresh": response_content.get("refresh"), "username": username})
                )
            return username, {"Authorization": f"Bearer {self._access}"}
        else:
            raise AuthenticationError(
                f"There was some issue logging in: {response.status_code} {response.text}"
            )

    def _refresh_token(self) -> Tuple[str, AuthenticationCredentials]:
        with open(os.path.join(Path.home(), ".seclea/config"), "r") as f:
            config = json.loads(f.read())
        try:
            refresh = config["refresh"]
            username = config["username"]
        except KeyError as e:
            print(e)
            # refresh token missing, prompt and login
            raise AuthenticationError
        response = self._transmission.send_json(
            url_path="/api/token/refresh/", obj={"refresh": refresh}
        )
        if not response.ok:
            if response.status_code == 401:
                print("Token has expired, please login.")
            else:
                handle_response(res=response, msg="There was an issue with the refresh token")
            raise AuthenticationError
        else:
            try:
                response_content = json.loads(response.content.decode("utf-8"))
                self._access = response_content.get("access")
                if self._access is not None:
                    return username, {"Authorization": f"Bearer {self._access}"}
                else:
                    raise AuthenticationError
            except Exception as e:
                print(e)
                raise AuthenticationError
