"""
Everything to do with the API to the backend.
"""
import json
from typing import Dict, List

import requests
from requests import Response

from seclea_ai.authentication import AuthenticationService
from seclea_ai.lib.seclea_utils.core.transmission import RequestWrapper


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

    def __init__(self, settings):
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
        self.auth.authenticate(self._transmission)
        self.project_endpoint = "/collection/projects"
        self.dataset_endpoint = "/collection/datasets"
        self.model_endpoint = "/collection/models"
        self.training_run_endpoint = "/collection/training-runs"
        self.model_states_endpoint = "/collection/model-states"

    def upload_dataset(
        self,
        dataset_file_path: str,
        project_pk: str,
        organization_pk: str,
        name: str,
        metadata: dict,
        dataset_hash: str,
        parent_dataset_hash: str = None,
        delete=False,
    ) -> Response:

        dataset_queryparams = {
            "project": str(project_pk),
            "organization": str(organization_pk),
            "name": name,
            "metadata": json.dumps(metadata),
            "hash": str(dataset_hash),
        }

        if parent_dataset_hash is not None:
            dataset_queryparams["parent"] = parent_dataset_hash

        res = self._transmission.send_file(
            url_path=f"{self.dataset_endpoint}",
            file_path=dataset_file_path,
            query_params=dataset_queryparams,
            delete_file=delete,
        )
        res = handle_response(response=res)

        return res

    def upload_model_state(
        self,
        model_state_file_path: str,
        organization_pk: str,
        project_pk: str,
        training_run_pk: str,
        sequence_num: int,
        final_state,
        delete=False,
    ):

        query_params = {
            "organization": organization_pk,
            "project": str(project_pk),
            "sequence_num": sequence_num,
            "training_run": str(training_run_pk),
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

    # TODO review typing.
    def upload_training_run(
        self,
        organization_pk: int,
        project_pk: int,
        dataset_pks: List[str],
        model_pk: int,
        training_run_name: str,
        params: Dict,
    ):
        data = {
            "organization": organization_pk,
            "project": project_pk,
            "datasets": dataset_pks,
            "model": model_pk,
            "name": training_run_name,
            "params": params,
        }

        res = self._transmission.send_json(
            url_path="/collection/training-runs",
            obj=data,
            query_params={"organization": organization_pk, "project": project_pk},
        )
        res = handle_response(response=res)
        return res

    def upload_transformation(
        self, name: str, code_raw, code_encoded, dataset_pk, organization, project
    ):

        data = {
            "name": name,
            "code_raw": code_raw,
            "code_encoded": code_encoded,
            "dataset": dataset_pk,
        }
        res = self._transmission.send_json(
            url_path="/collection/dataset-transformations",
            obj=data,
            query_params={"organization": organization, "project": project},
        )
        res = handle_response(response=res)
        return res

    def update_dataset_metadata(self, dataset_hash, metadata, organization, project):
        """
        Update the dataset's metadata. For use when the metadata is too large to encode in the url.
        @param dataset_hash:
        @param metadata:
        @return:
        """
        res = self._transmission.patch(
            url_path=f"/collection/datasets/{dataset_hash}",
            obj={"metadata": metadata},
            query_params={"organization": organization, "project": project},
        )
        res = handle_response(response=res)

        return res
