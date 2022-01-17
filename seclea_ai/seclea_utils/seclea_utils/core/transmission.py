from abc import ABC
from typing import Any, Dict

import requests
import ujson as json
from requests import Response


# Interface declaration #
class Transmission(ABC):
    def __init__(self, server_root_url):
        self._server_root = server_root_url
        self._headers = {}
        self._cookies = {}

    @property
    def cookies(self) -> dict:
        return self._cookies.copy()

    @cookies.setter
    def cookies(self, new_cookies: Dict) -> None:
        self._cookies.update(new_cookies)
        self.headers = {"HTTP_COOKIE": json.dumps(self.cookies)}

    @property
    def headers(self) -> Dict:
        """
        Returns a copy of the headers.
        :return: Dict A copy of the headers
        """
        return self._headers.copy()

    @headers.setter
    def headers(self, new_headers: Dict) -> None:
        """
        Adds new headers to the classes stored headers.
        :param new_headers: dict The new headers to add
        :return: None
        """
        self._headers = {**self._headers, **new_headers}

    def load_file(self, url_path: str, file_path: str, query_params: Dict = None) -> None:
        """
        Loads a file from an endpoint to a specified file path.
        """
        pass

    def send_file(self, url_path: str, file_path: str, query_params: Dict = None) -> Any:
        """
        Sends a file from a specified path to an endpoint.
        """
        pass

    def get(self, url_path: str, query_params: Dict = None) -> Any:
        """
        Sends a get request to a specified endpoint.
        """
        pass

    def send_json(self, url_path: str, obj: Dict, query_params: Dict = None) -> Any:
        """
        Sends a json object to a specified endpoint.
        """
        pass


# Start of concrete implementations #
class RequestWrapper(Transmission):
    def __init__(self, server_root_url):
        super(RequestWrapper, self).__init__(server_root_url)

    def load_file(self, url_path: str, file_path: str, query_params: Dict = None) -> None:
        res = requests.get(self._server_root + url_path, headers=self.headers, params=query_params)
        if res.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(res.content)
        else:
            raise Exception(
                f"Issue requesting file, status: {res.status_code}, Reason: {res.reason} {res.text}"
            )

    def send_file(self, url_path: str, file_path: str, query_params: Dict = None) -> Response:
        with open(file_path, "rb") as f:
            headers = self.headers
            headers["Content-Disposition"] = f"attachment; filename={url_path}"
            request_path = f"{self._server_root}{url_path}"
            return requests.post(
                request_path,
                files={"file": f},
                headers=headers,
                cookies=self.cookies,
                params=query_params,
            )

    def get(self, url_path: str, query_params: Dict = None) -> Response:
        request_path = f"{self._server_root}{url_path}"
        return requests.get(
            request_path, headers=self.headers, params=query_params, cookies=self.cookies
        )

    def send_json(self, url_path: str, obj: Dict, query_params: Dict = None) -> Response:
        headers = self.headers
        headers["content-type"] = "application/json"
        request_path = f"{self._server_root}{url_path}"
        return requests.post(
            request_path,
            data=json.dumps(obj),
            headers=headers,
            cookies=self.cookies,
            params=query_params,
        )
