"""
Everything to do with the API to the backend.
"""
import json
import logging
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
    ServiceDegradedError,
    ImATeapotError,
)
from ...internal.authentication import AuthenticationService


def handle_response(response: Response, msg: str = ""):
    if response.status_code in [200, 201]:  # or requests.code.ok
        return response
    err_msg = f"{response.status_code} - {response.reason} \n{msg} - {response.text}"
    if response.status_code == 400:
        raise BadRequestError(err_msg)
    if response.status_code == 401:
        raise AuthenticationError(err_msg)
    if response.status_code == 403:
        raise AuthorizationError(err_msg)
    if response.status_code == 404:
        raise NotFoundError(err_msg)
    if response.status_code == 418:
        raise ImATeapotError(err_msg)
    if response.status_code in {500, 502, 503, 504}:
        raise ServiceDegradedError(err_msg)
    if str(response.status_code).startswith("5"):
        raise ServerError(err_msg)
    raise APIError(err_msg)


class Api:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings, username=None, password=None):
        # setup some defaults
        self._settings = settings
        self._session = requests.Session()
        self.auth = AuthenticationService(url=settings["auth_url"])
        # TODO maybe remove auth on creation - only when needed?
        self.auth.authenticate(self._session, username=username, password=password)
        self._root_url = settings["platform_url"]
        self._project_endpoint = "collection/projects"
        self._dataset_endpoint = "collection/datasets"
        self._dataset_transformations_endpoint = "collection/dataset-transformations"
        self._model_endpoint = "collection/models"
        self._training_run_endpoint = "collection/training-runs"
        self._model_states_endpoint = "collection/model-states"

    def __del__(self):
        self._session.close()

    def authenticate(self, username=None, password=None):
        self.auth.authenticate(session=self._session, username=username, password=password)

    @staticmethod
    def test_json_valid(d):
        d = json.dumps(d)
        json.loads(d)
        pass

    def get_project(self, project_id, organization_id, **filter_kwargs) -> Dict:
        return handle_response(
            self._session.get(
                url=f"{self._root_url}/{self._project_endpoint}/{project_id}",
                params={"organization": organization_id, **filter_kwargs},
            )
        ).json()

    def get_projects(self, organization_id, **filter_kwargs) -> List:
        return handle_response(
            self._session.get(
                url=f"{self._root_url}/{self._project_endpoint}",
                params={"organization": organization_id, **filter_kwargs},
            )
        ).json()

    def upload_project(self, name, description, organization_id) -> Dict:
        return handle_response(
            self._session.post(
                url=f"{self._root_url}/{self._project_endpoint}",
                json={
                    "name": name,
                    "description": description,
                    "organization": organization_id,
                },
                params={"organization": organization_id},
            )
        ).json()

    def get_dataset(self, dataset_id: str, project_id, organization_id) -> Dict:
        return handle_response(
            self._session.get(
                url=f"{self._root_url}/{self._dataset_endpoint}/{dataset_id}",
                params={"project": project_id, "organization": organization_id},
            )
        ).json()

    def upload_dataset(
        self,
        dataset_file_path: str,
        project_id: str,
        organization_id: str,
        name: str,
        metadata: dict,
        dataset_id: int,
        parent_dataset_id: str = None,
    ) -> Dict:

        dataset_queryparams = {"project": project_id, "organization": organization_id}
        self.test_json_valid(metadata)

        with open(dataset_file_path, "rb") as f:
            dataset_obj = {
                "project": (None, project_id),
                "name": (None, name),
                "metadata": (None, json.dumps(metadata), "application/json"),
                "hash": (None, str(dataset_id)),
                "dataset": (os.path.basename(dataset_file_path), f),
            }
            if parent_dataset_id is not None:
                dataset_obj["parent"] = (None, parent_dataset_id)
                logging.debug(f"dataset_obj dataset field: {dataset_obj['parent']}")

            return handle_response(
                self._session.post(
                    url=f"{self._root_url}/{self._dataset_endpoint}",
                    files=dataset_obj,
                    params=dataset_queryparams,
                )
            ).json()

    def get_models(self, project_id, organization_id, **filter_kwargs) -> List:
        """
        Get models - with optional filter parameters.

        :param project_id:

        :param organization_id:

        :param filter_kwargs: Optional filter parameters
            Available filter params are:
            - name: str The name of the model
            - framework: str The name of the framework of the model - see XXX for options

        :return: Response
        """
        return handle_response(
            self._session.get(
                url=f"{self._root_url}/{self._model_endpoint}",
                params={"organization": organization_id, "project": project_id, **filter_kwargs},
            )
        ).json()

    def upload_model(self, organization_id, project_id, model_name, framework_name) -> Dict:
        return handle_response(
            self._session.post(
                url=f"{self._root_url}/{self._model_endpoint}",
                json={
                    "organization": organization_id,
                    "project": project_id,
                    "name": model_name,
                    "framework": framework_name,
                },
                params={"organization": organization_id, "project": project_id},
            )
        ).json()

    # TODO review if use of id is confusing - may need to standardise id params
    def get_training_runs(self, project_id: int, organization_id: str, **filter_kwargs) -> List:
        return handle_response(
            self._session.get(
                url=f"{self._root_url}/{self._training_run_endpoint}",
                params={
                    "project": project_id,
                    "organization": organization_id,
                    **filter_kwargs,
                },
            )
        ).json()

    # TODO review typing.
    def upload_training_run(
        self,
        organization_id: int,
        project_id: int,
        dataset_ids: List[str],
        model_id: int,
        training_run_name: str,
        params: Dict,
    ) -> Dict:
        data = {
            "organization": organization_id,
            "project": project_id,
            "datasets": dataset_ids,
            "model": model_id,
            "name": training_run_name,
            "params": params,
        }
        self.test_json_valid(data)

        return handle_response(
            self._session.post(
                url=f"{self._root_url}/{self._training_run_endpoint}",
                json=data,
                params={"organization": organization_id, "project": project_id},
            )
        ).json()

    def upload_model_state(
        self,
        model_state_file_path: str,
        organization_id: str,
        project_id: str,
        training_run_id: int,
        sequence_num: int,
        final_state,
    ) -> Dict:
        with open(model_state_file_path, "rb") as f:
            return handle_response(
                self._session.post(
                    url=f"{self._root_url}/{self._model_states_endpoint}",
                    files={
                        "project": (None, project_id),
                        "sequence_num": (None, sequence_num),
                        "training_run": (None, training_run_id),
                        "final_state": (None, final_state),
                        "state": (os.path.basename(model_state_file_path), f),
                    },
                    params={
                        "organization": organization_id,
                        "project": project_id,
                    },
                )
            ).json()

    def get_transformations(self, project_id, organization_id, **filter_kwargs) -> List:
        return handle_response(
            self._session.get(
                url=f"{self._root_url}/{self._dataset_transformations_endpoint}",
                params={
                    "project": project_id,
                    "organization": organization_id,
                    **filter_kwargs,
                },
            )
        ).json()

    def upload_transformation(
        self, name: str, code_raw, code_encoded, dataset_id: int, organization_id, project_id
    ) -> Dict:

        data = {
            "name": name,
            "code_raw": code_raw,
            "code_encoded": code_encoded,
            "dataset": dataset_id,
        }
        return handle_response(
            self._session.post(
                url=f"{self._root_url}/{self._dataset_transformations_endpoint}",
                json=data,
                params={"organization": organization_id, "project": project_id},
            )
        ).json()
