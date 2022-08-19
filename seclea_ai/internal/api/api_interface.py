"""
Everything to do with the API to the backend.
"""
import json
import os
from typing import Dict, List

import requests
from requests import Response

from ..exceptions import (
    BadRequestError,
    AuthenticationError,
    AuthorizationError,
    APIError,
    NotFoundError,
    ServerError,
)
from ...internal.authentication import AuthenticationService
from seclea_ai.lib.seclea_utils.object_management.mixin import ProjectMixin
from abc import ABC


def handle_response(response: Response, msg: str = ""):
    if response.status_code in [200, 201]:  # or requests.code.ok
        return response
    err_msg = f"{response.status_code} Error \n{msg} - {response.reason} - {response.text}"
    if response.status_code == 400:
        raise BadRequestError(
            f"400 Error - Bad Request\n +{msg} - {response.reason} - {response.text}"
        )
    if response.status_code == 401:
        raise AuthenticationError(
            f"401 Error - Unauthorized\n +{msg} - {response.reason} - {response.text}"
        )
    if response.status_code == 403:
        raise AuthorizationError(
            f"403 Error - Forbidden\n +{msg} - {response.reason} - {response.text}"
        )
    if response.status_code == 404:
        raise NotFoundError(f"404 Error - Not Found\n +{msg} - {response.reason} - {response.text}")
    if response.status_code == 500:
        raise ServerError(
            f"500 Error - Internal Server Error\n +{msg} - {response.reason} - {response.text}"
        )

    raise APIError(err_msg)


class BaseApi(ABC):
    _session: requests.Session
    _url_root:str


class BaseAppApi(BaseApi):
    _url_app: str


class BaseModelApi(BaseAppApi):
    """
    Minimum api functionality required by any object on the back-end
    """
    _url_model: str

    def get_list(self, *args, **kwargs):
        pass

    def get(self, *args, **kwargs):
        """
        Given
        @param arga:
        @param kwargs:aaaaaa
        @return:
        """
        pass

    def delete(self, *args, **kwargs):
        pass

    def patch(self, *args, **kwargs):
        pass

class PlatformApi:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, platform_url, auth_url, username=None, password=None):
        # setup some defaults
        self._session = requests.Session()
        self.auth = AuthenticationService(url=auth_url)
        # TODO maybe remove auth on creation - only when needed?
        self.auth.authenticate(self._session, username=username, password=password)
        self._root_url = platform_url
        self._project_endpoint = "collection/projects"
        self._organization_endpoint = "organization/"
        self._dataset_endpoint = "collection/datasets"
        self._dataset_transformations_endpoint = "collection/dataset-transformations"
        self._model_endpoint = "collection/models"
        self._training_run_endpoint = "collection/training-runs"
        self._model_states_endpoint = "collection/model-states"

    def __del__(self):
        self._session.close()

