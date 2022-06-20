"""
Everything to do with the API to the backend.
"""
import json
from typing import Dict, List

import requests
from requests import Response

from seclea_ai.authentication import AuthenticationService
from seclea_ai.lib.seclea_utils.core.transmission import RequestWrapper


# TODO return Exceptions for specific non success responses.
def handle_response(response: Response, msg: str = ""):
    if response.status_code in [200, 201]:  # or requests.code.ok
        return response
    err_msg = f"{response.status_code} Error \n{msg} - {response.reason} - {response.text}"
    if response.status_code == 400:
        err_msg = f"400 Error - Bad Request\n +{msg} - {response.reason} - {response.text}"
    if response.status_code == 401:
        err_msg = f"401 Error - Unauthorized\n +{msg} - {response.reason} - {response.text}"
    if response.status_code == 403:
        err_msg = f"403 Error - Forbidden\n +{msg} - {response.reason} - {response.text}"
    if response.status_code == 500:
        err_msg = (
            f"500 Error - Internal Server Error\n +{msg} - {response.reason} - {response.text}"
        )

    raise ValueError(err_msg)


class Api:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings, username=None, password=None):
        # setup some defaults
        self._settings = settings
        self.transport = requests.Session()
        self._transmission = RequestWrapper(
            server_root_url=settings["platform_url"]
        )  # TODO replace
        self.auth = AuthenticationService(
            url=settings["auth_url"],
            transmission=RequestWrapper(server_root_url=settings["auth_url"]),
        )
        # TODO maybe remove auth on creation - only when needed?
        self.auth.authenticate(self._transmission, username=username, password=password)
        self.project_endpoint = "/collection/projects"
        self.dataset_endpoint = "/collection/datasets"
        self.model_endpoint = "/collection/models"
        self.training_run_endpoint = "/collection/training-runs"
        self.model_states_endpoint = "/collection/model-states"

    def authenticate(self, username=None, password=None):
        self.auth.authenticate(
            transmission=self._transmission, username=username, password=password
        )

    def get_projects(self, organization_id, **filter_kwargs) -> Response:
        res = self._transmission.get(
            url_path="/collection/projects",
            query_params={"organization": organization_id, **filter_kwargs},
        )
        res = handle_response(response=res)
        return res

    def upload_project(self, name, description, organization_id) -> Response:
        res = self._transmission.send_json(
            url_path="/collection/projects",
            obj={
                "name": name,
                "description": description,
                "organization": organization_id,
            },
            query_params={"organization": organization_id},
        )
        res = handle_response(response=res)
        return res

    def get_dataset(self, dataset_id: str, project_id, organization_id) -> Response:
        res = self._transmission.get(
            url_path=f"/collection/datasets/{dataset_id}",
            query_params={"project": project_id, "organization": organization_id},
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
        dataset_id: str,
        parent_dataset_id: str = None,
        delete=False,
    ) -> Response:

        dataset_queryparams = {
            "project": str(project_id),
            "organization": organization_id,
            "name": name,
            "metadata": json.dumps(metadata),
            "hash": str(dataset_id),
        }

        if parent_dataset_id is not None:
            dataset_queryparams["parent"] = parent_dataset_id

        res = self._transmission.send_file(
            url_path=f"{self.dataset_endpoint}",
            file_path=dataset_file_path,
            query_params=dataset_queryparams,
            delete_file=delete,
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
        res = self._transmission.get(
            url_path="/collection/models",
            query_params={"organization": organization_id, "project": project_id, **filter_kwargs},
        )
        res = handle_response(response=res)
        return res

    def upload_model(self, organization_id, project_id, model_name, framework_name) -> Response:
        res = self._transmission.send_json(
            url_path="/collection/models",
            obj={
                "organization": organization_id,
                "project": project_id,
                "name": model_name,
                "framework": framework_name,
            },
            query_params={"organization": organization_id, "project": project_id},
        )
        res = handle_response(response=res)
        return res

    # TODO review if use of id is confusing - may need to standardise id params
    def get_training_runs(self, project_id: int, organization_id: str, **filter_kwargs) -> Response:
        res = self._transmission.get(
            "/collection/training-runs",
            query_params={
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

        res = self._transmission.send_json(
            url_path="/collection/training-runs",
            obj=data,
            query_params={"organization": organization_id, "project": project_id},
        )
        res = handle_response(response=res)
        return res

    def upload_model_state(
        self,
        model_state_file_path: str,
        organization_id: str,
        project_id: str,
        training_run_id: str,
        sequence_num: int,
        final_state,
        delete=False,
    ):

        query_params = {
            "organization": organization_id,
            "project": str(project_id),
            "sequence_num": sequence_num,
            "training_run": str(training_run_id),
            "final_state": str(final_state),
        }

        res = self._transmission.send_file(
            url_path=f"{self.model_states_endpoint}",
            file_path=model_state_file_path,
            query_params=query_params,
            delete_file=delete,
        )
        res = handle_response(response=res)

        return res

    def upload_transformation(
        self, name: str, code_raw, code_encoded, dataset_id, organization_id, project_id
    ):

        data = {
            "name": name,
            "code_raw": code_raw,
            "code_encoded": code_encoded,
            "dataset": dataset_id,
        }
        res = self._transmission.send_json(
            url_path="/collection/dataset-transformations",
            obj=data,
            query_params={"organization": organization_id, "project": project_id},
        )
        res = handle_response(response=res)
        return res

    def update_dataset_metadata(self, dataset_id, metadata, organization_id, project_id):
        res = self._transmission.patch(
            url_path=f"/collection/datasets/{dataset_id}",
            obj={"metadata": metadata},
            query_params={"organization": organization_id, "project": project_id},
        )
        res = handle_response(response=res)

        return res
