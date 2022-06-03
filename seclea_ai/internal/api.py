"""
Everything to do with the API to the backend.
"""
import requests
import asyncio
import aiohttp
import json
import os
import inspect

from requests import Response
from seclea_ai.authentication import AuthenticationService
from seclea_ai.transformations import DatasetTransformation
from seclea_ai.lib.seclea_utils.core import (
    encode_func,
)
from seclea_ai.lib.seclea_utils.core.transmission import Transmission

def handle_response(res: Response, expected: int, msg: str) -> Response:
    if not res.status_code == expected:
        raise ValueError(
            f"Response Status code {res.status_code}, expected:{expected}. \n{msg} - {res.reason} - {res.text}"
        )
    return res

class Api:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings):
        # setup some defaults
        self._settings = settings
        self.transport = requests.Session()
        self.auth = AuthenticationService(url=settings["auth_url"], session=self.transport)
        self.auth.authenticate()
        self.project_endpoint = "/collection/projects"
        self.dataset_endpoint = "/collection/datasets"
        self.model_endpoint = "/collection/models"
        self.training_run_endpoint = "/collection/training-runs"
        self.model_states_endpoint = "/collection/model-states"

    def reauthenticate(self):
        if not self.auth.verify_token(transmission=self.transport) and not self.auth.refresh_token(
            transmission=self.transport
        ):
            self.auth.authenticate(transmission=self.transport)

    def post_dataset(self,
            transmission: Transmission,
            dataset_file_path: str,
            project_pk: str,
            organization_pk: str,
            name: str,
            metadata: dict,
            dataset_hash: str,
            parent_dataset_hash: str = None,
            delete=False,
    ):

        dataset_queryparams = {
            "project": str(project_pk),
            "organization": str(organization_pk),
            "name": name,
            "metadata": json.dumps(metadata),
            "hash": str(dataset_hash),
        }

        if parent_dataset_hash is not None:
            dataset_queryparams['parent'] = parent_dataset_hash

        res = asyncio.run(self.upload_to_server(f"{self.dataset_endpoint}", dataset_file_path, dataset_queryparams, transmission, delete))

        return res

    def post_model_state(self,
            transmission: Transmission,
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

        res = asyncio.run(self.upload_to_server(f"{self.model_states_endpoint}", model_state_file_path, query_params, transmission, delete))

        return res

    async def upload_to_server(self, url_path, dataset_file_path, queryparams, transmission, delete_file):
        """
        send file to server
        """
        with open(dataset_file_path, 'rb') as f:
            headers = transmission.headers
            cookies = transmission.cookies
            headers["Content-Disposition"] = f"attachment; filename={url_path}"
            request_path = f"{transmission._server_root}{url_path}"

            try:
                async with aiohttp.ClientSession(cookies=cookies, headers=headers) as session:
                    async with session.post(request_path, data={'file': f}, params=queryparams) as response:
                        if delete_file:
                            os.remove(dataset_file_path)
                        return response
            except Exception as e:
                print(e)

    def _upload_transformation(self, transformation: DatasetTransformation, dataset_pk, transmission, organization, project):
        idx = 0
        trans_kwargs = {**transformation.data_kwargs, **transformation.kwargs}
        data = {
            "name": transformation.func.__name__,
            "code_raw": inspect.getsource(transformation.func),
            "code_encoded": encode_func(transformation.func, [], trans_kwargs),
            "order": idx,
            "dataset": dataset_pk,
        }
        res = transmission.send_json(
            url_path="/collection/dataset-transformations",
            obj=data,
            query_params={"organization": organization, "project": project},
        )

        res = handle_response(
            res,
            expected=201,
            msg=f"There was an issue uploading the transformations on transformation {idx} with name {transformation.func.__name__}: {res.text}",
        )
        return res

    def _update_dataset_metadata(self, dataset_hash, metadata, transmission, organization, project):
        """
        Update the dataset's metadata. For use when the metadata is too large to encode in the url.
        @param dataset_hash:
        @param metadata:
        @return:
        """
        res = transmission.patch(
            url_path=f"/collection/datasets/{dataset_hash}",
            obj={
                "metadata": metadata,
            },
            query_params={"organization": organization, "project": project},
        )

        return handle_response(
            res, expected=200, msg=f"There was an issue updating the metadata: {res.text}"
        )