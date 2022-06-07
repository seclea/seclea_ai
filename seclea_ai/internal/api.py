"""
Everything to do with the API to the backend.
"""
import asyncio
import inspect
import json
import os
from typing import Dict

import aiohttp
import requests
from requests import Response

from seclea_ai.lib.seclea_utils.core import encode_func
from seclea_ai.lib.seclea_utils.core.transmission import Transmission
from seclea_ai.transformations import DatasetTransformation


def handle_response(response: Response, msg: str = ""):
    if response.status_code in [200, 201]:  # or requests.code.ok
        return response
    err_msg = f"{response.status_code} Error \n{msg} - {response.reason} - {response.text}"
    if response.status_code == 400:
        err_msg = f"400 Error - Bad Request\n +{msg} - {response.reason} - {response.text}"
    if response.status_code == 401:
        err_msg = f"401 Error - Unauthorized\n +{msg} - {response.reason} - {response.text}"
    if response.status_code == 500:
        err_msg = (
            f"500 Error - Internal Server Error\n +{msg} - {response.reason} - {response.text}"
        )
    if response.status_code == 500:
        err_msg = (
            f"502 Error - Internal Server Error\n +{msg} - {response.reason} - {response.text}"
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
        # self.auth = AuthenticationService(url=settings["auth_url"], session=self.transport)
        # self.auth.authenticate()
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

    def post_dataset(
        self,
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
            dataset_queryparams["parent"] = parent_dataset_hash

        res = asyncio.run(
            self.send_file(
                f"{self.dataset_endpoint}",
                dataset_file_path,
                transmission,
                dataset_queryparams,
                delete,
            )
        )

        return res

    def post_model_state(
        self,
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

        res = asyncio.run(
            self.send_file(
                f"{self.model_states_endpoint}",
                model_state_file_path,
                transmission,
                query_params,
                delete,
            )
        )

        return res

    def _upload_transformation(
        self, transformation: DatasetTransformation, dataset_pk, transmission, organization, project
    ):
        idx = 0
        trans_kwargs = {**transformation.data_kwargs, **transformation.kwargs}
        data = {
            "name": transformation.func.__name__,
            "code_raw": inspect.getsource(transformation.func),
            "code_encoded": encode_func(transformation.func, [], trans_kwargs),
            "order": idx,
            "dataset": dataset_pk,
        }
        res = asyncio.run(
            self.send_json(
                url_path="/collection/dataset-transformations",
                obj=data,
                query_params={"organization": organization, "project": project},
                transmission=transmission,
            )
        )
        return res

    def _update_dataset_metadata(self, dataset_hash, metadata, transmission, organization, project):
        """
        Update the dataset's metadata. For use when the metadata is too large to encode in the url.
        @param dataset_hash:
        @param metadata:
        @return:
        """
        res = asyncio.run(
            self.patch(
                url_path=f"/collection/datasets/{dataset_hash}",
                obj={
                    "metadata": metadata,
                },
                query_params={"organization": organization, "project": project},
                transmission=transmission,
            )
        )

        return res

    async def send_file(
        self,
        url_path: str,
        file_path: str,
        transmission,
        query_params: Dict = None,
        delete_file=False,
        json_response=False,
    ) -> aiohttp.ClientResponse:
        """
        send file to server
        """
        with open(file_path, "rb") as f:
            headers = transmission.headers
            cookies = transmission.cookies
            headers["Content-Disposition"] = f"attachment; filename={url_path}"
            request_path = f"{transmission._server_root}{url_path}"
            async with aiohttp.ClientSession(cookies=cookies, headers=headers) as session:
                async with session.post(
                    request_path, data={"file": f}, params=query_params
                ) as response:
                    response = await self._wrap_response(response, json_response)
                    if response and delete_file:
                        os.remove(file_path)
                    return response

    async def send_json(
        self, url_path: str, obj: Dict, transmission, query_params: Dict = None, json_response=False
    ) -> aiohttp.ClientResponse:
        """
        Send json data to server
        """
        cookies = transmission.cookies
        headers = transmission.headers
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
        request_path = f"{transmission._server_root}{url_path}"
        async with aiohttp.ClientSession(
            cookies=cookies, headers=headers, cookie_jar=None
        ) as session:
            async with session.post(
                request_path, data=json.dumps(obj), params=query_params
            ) as response:
                return await self._wrap_response(response, json_response)

    async def patch(
        self, url_path: str, obj: Dict, transmission, query_params: Dict = None, json_response=False
    ) -> aiohttp.ClientResponse:
        cookies = transmission.cookies
        headers = transmission.headers
        headers["content-type"] = "application/json"
        request_path = f"{transmission._server_root}{url_path}"
        async with aiohttp.ClientSession(cookies=cookies, headers=headers) as session:
            async with session.patch(
                request_path, data=json.dumps(obj), params=query_params
            ) as response:
                return await self._wrap_response(response, json_response)

    async def _wrap_response(self, response: aiohttp.ClientResponse, json_response=False):
        if response.status in [200, 201]:  # or requests.code.ok
            if json_response:
                return await response.json()
            return response
        err_msg = f"{response.status} Error - {response.reason} - {response.text}"
        if response.status == 400:
            err_msg = f"400 Error - Bad Request\n + - {response.reason} - {response.text}"
        if response.status == 401:
            err_msg = f"401 Error - Unauthorized\n + - {response.reason} - {response.text}"
        if response.status == 500:
            err_msg = f"500 Error - Internal Server Error\n + - {response.reason} - {response.text}"
        if response.status == 500:
            err_msg = f"502 Error - Internal Server Error\n + - {response.reason} - {response.text}"

        raise ValueError(err_msg)
