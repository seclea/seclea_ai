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
from seclea_ai.internal.mixins import APIMixin, ProjectMixin, OrganizationMixin


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

    def get_project(self, project: ProjectMixin, **filter_kwargs) -> Response:
        return handle_response(
            self._session.get(
                url=f"{self._root_url}/{self._project_endpoint}/{project.uuid}",
                params={"organization": project.organization.uuid, **filter_kwargs},
            )
        )

    def get_projects(self, organization_id, **filter_kwargs) -> Response:
        res = self._session.get(
            url=f"{self._root_url}/{self._project_endpoint}",
            params={"organization": organization_id, **filter_kwargs},
        )
        res = handle_response(response=res)
        return res

    def upload_project(self, project: ProjectMixin) -> Response:
        res = self._session.post(
            url=f"{self._root_url}/{self._project_endpoint}",
            json=project.serialize(),
            params={"organization": project.organization.uuid},
        )
        res = handle_response(response=res)
        return res

    def get_dataset(self, dataset_id: str, project_id, organization_id) -> Response:
        res = self._session.get(
            url=f"{self._root_url}/{self._dataset_endpoint}/{dataset_id}",
            params={"project": project_id, "organization": organization_id},
        )
        res = handle_response(response=res)
        return res

    def upload_dataset(
            self,
            dataset_file_path: str,
            project_id: str,
            organization_id: str,
            name: str,
            metadata: dict,
            dataset_id: int,
            parent_dataset_id: str = None,
    ) -> Response:
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
                print(f"dataset_obj dataset field: {dataset_obj['parent']}")

            res = self._session.post(
                url=f"{self._root_url}/{self._dataset_endpoint}",
                files=dataset_obj,
                params=dataset_queryparams,
            )
        res = handle_response(response=res)
        return res

    def get_models(self, project_id, organization_id, **filter_kwargs) -> Response:
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
        res = self._session.get(
            url=f"{self._root_url}/{self._model_endpoint}",
            params={"organization": organization_id, "project": project_id, **filter_kwargs},
        )
        res = handle_response(response=res)
        return res

    def upload_model(self, organization_id, project_id, model_name, framework_name) -> Response:
        res = self._session.post(
            url=f"{self._root_url}/{self._model_endpoint}",
            json={
                "organization": organization_id,
                "project": project_id,
                "name": model_name,
                "framework": framework_name,
            },
            params={"organization": organization_id, "project": project_id},
        )
        res = handle_response(response=res)
        return res

    # TODO review if use of id is confusing - may need to standardise id params
    def get_training_runs(self, project_id: int, organization_id: str, **filter_kwargs) -> Response:
        res = self._session.get(
            url=f"{self._root_url}/{self._training_run_endpoint}",
            params={
                "project": project_id,
                "organization": organization_id,
                **filter_kwargs,
            },
        )
        res = handle_response(response=res)
        return res

    # TODO review typing.
    def upload_training_run(
            self,
            organization_id: int,
            project_id: int,
            dataset_ids: List[str],
            model_id: int,
            training_run_name: str,
            params: Dict,
    ):
        data = {
            "organization": organization_id,
            "project": project_id,
            "datasets": dataset_ids,
            "model": model_id,
            "name": training_run_name,
            "params": params,
        }
        self.test_json_valid(data)

        res = self._session.post(
            url=f"{self._root_url}/{self._training_run_endpoint}",
            json=data,
            params={"organization": organization_id, "project": project_id},
        )
        res = handle_response(response=res)
        return res

    def upload_model_state(
            self,
            model_state_file_path: str,
            organization_id: str,
            project_id: str,
            training_run_id: int,
            sequence_num: int,
            final_state,
    ):
        with open(model_state_file_path, "rb") as f:
            res = self._session.post(
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
        res = handle_response(response=res)
        return res

    def upload_transformation(
            self, name: str, code_raw, code_encoded, dataset_id: int, organization_id, project_id
    ):
        data = {
            "name": name,
            "code_raw": code_raw,
            "code_encoded": code_encoded,
            "dataset": dataset_id,
        }
        res = self._session.post(
            url=f"{self._root_url}/{self._dataset_transformations_endpoint}",
            json=data,
            params={"organization": organization_id, "project": project_id},
        )
        res = handle_response(response=res)
        return res
