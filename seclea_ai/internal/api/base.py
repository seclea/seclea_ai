import abc
from typing import List, Set

from requests import Response
from requests import Session

from seclea_ai.lib.seclea_utils.object_management.mixin import BaseModel
from .exceptions import API_ERROR_CODE_EXCEPTION_MAPPER, ApiError
from .status import HTTP_201_CREATED, HTTP_200_OK
import json


class BaseModelApi:

    def __init__(self, base_url: str, session: Session):
        """

        @param base_url:
        @param session:
        """
        self.url = f'{base_url}{self.model_url}'
        self.session = session

    @property
    @abc.abstractmethod
    def model_url(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def file_keys(self) -> Set[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def json_keys(self) -> Set[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def model(self) -> BaseModel.__class__:
        raise NotImplementedError

    @staticmethod
    def process_response(resp: Response, expected_code=HTTP_200_OK):
        print('processing response: ', resp.status_code, resp.reason, resp.text[:20])
        if resp.status_code != expected_code:
            err_cls = API_ERROR_CODE_EXCEPTION_MAPPER[resp.status_code]
            raise err_cls(resp=resp)

    def get_list(self, params: dict) -> List[BaseModel]:
        resp = self.session.get(url=self.url + '/', params=params)
        self.process_response(resp)
        return [self.model(**d) for d in resp.json()]

    def get(self, pk: str, params: dict = None) -> BaseModel:
        """

        @param pk:
        @param params:
        @return:
        """
        resp = self.session.get(url=f'{self.url}/{pk}', params=params)
        self.process_response(resp)
        return self.model(**resp.json())

    def get_one_from_list(self, **params: dict) -> BaseModel:
        result = self.get_list(params=params)
        if len(result) == 1:
            return result[0]
        raise Exception(f'Incorrect number of arguments returned, filter: {params} result: {result}')

    def delete(self, pk, params: dict) -> BaseModel:
        resp = self.session.delete(url=f'{self.url}/{pk}', params=params)
        self.process_response(resp)
        return self.model(**resp.json())

    def patch(self, pk, patch_data: dict, params: dict) -> BaseModel:
        resp = self.session.patch(url=f'{self.url}/{pk}', params=params, data=patch_data)
        self.process_response(resp)
        return self.model(**resp.json())

    def create(self, create_data: dict, params: dict) -> BaseModel:
        """
        @param create_data:
        @param params:
        @return:
        """
        # temporary generic file upload workaround.
        print('create object initiated: ', params)
        if len(self.file_keys) > 0:
            files = []  # store any opened files for cleanup after response.
            for key, val in create_data.items():
                if key in self.file_keys:
                    f = open(create_data[key], 'rb')
                    files.append(f)
                    create_data[key] = (key, f)
                elif key in self.json_keys:
                    create_data[key] = (None, json.dumps(val), 'application/json')
                else:
                    create_data[key] = (None, create_data[key])

            print("sending request: ", create_data.keys())
            resp = self.session.post(url=f'{self.url}', params=params, files=create_data)
            for f in files:
                f.close()
        else:
            print("sending request: ", create_data)
            resp = self.session.post(url=f'{self.url}', params=params, data=create_data)
        print("create object complete")
        self.process_response(resp, expected_code=HTTP_201_CREATED)
        return self.model(**resp.json())
    # def upload_dataset(
    #     self,
    #     dataset_file_path: str,
    #     project_id: str,
    #     organization_id: str,
    #     name: str,
    #     metadata: dict,
    #     dataset_id: int,
    #     parent_dataset_id: str = None,
    # ) -> Dict:
    #
    #     dataset_queryparams = {"project": project_id, "organization": organization_id}
    #     self.test_json_valid(metadata)
    #
    #     with open(dataset_file_path, "rb") as f:
    #         dataset_obj = {
    #             "project": (None, project_id),
    #             "name": (None, name),
    #             "metadata": (None, json.dumps(metadata), "application/json"),
    #             "hash": (None, str(dataset_id)),
    #             "dataset": (os.path.basename(dataset_file_path), f),
    #         }
    #         if parent_dataset_id is not None:
    #             dataset_obj["parent"] = (None, parent_dataset_id)
    #             logging.debug(f"dataset_obj dataset field: {dataset_obj['parent']}")
    #
    #         return handle_response(
    #             self._session.post(
    #                 url=f"{self._root_url}/{self._dataset_endpoint}",
    #                 files=dataset_obj,
    #                 params=dataset_queryparams,
    #             )
    #         ).json()
