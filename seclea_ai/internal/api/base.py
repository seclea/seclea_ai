import abc
from typing import List

from requests import Response
from requests import Session

from seclea_ai.lib.seclea_utils.object_management.mixin import BaseModel
from .exceptions import API_ERROR_CODE_EXCEPTION_MAPPER
from .status import HTTP_201_CREATED, HTTP_200_OK


class BaseModelApi:
    def __init__(self, base_url: str, session: Session):
        """

        @param base_url:
        @param session:
        """
        self.url = f'{base_url}{self.model_url}'
        self.session = session

    @abc.abstractmethod
    @property
    def model_url(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    @property
    def model(self) -> BaseModel.__class__:
        raise NotImplementedError

    @staticmethod
    def process_response(resp: Response, expected_code=HTTP_200_OK):
        if resp.status_code != expected_code:
            raise API_ERROR_CODE_EXCEPTION_MAPPER[resp.status_code](resp)

    def get_list(self, params: dict) -> List[BaseModel]:
        resp = self.session.get(url=self.url, params=params)
        self.process_response(resp)
        return [self.model().deserialize(d) for d in resp.json()]

    def get(self, pk: str, params: dict) -> BaseModel:
        """

        @param pk:
        @param params:
        @return:
        """
        resp = self.session.get(url=f'{self.url}{pk}', params=params)
        self.process_response(resp)
        return self.model().deserialize(resp.json())

    def get_one_from_list(self, params: dict) -> BaseModel:
        result = self.get_list(params=params)
        if len(result) == 1:
            return result[0]
        raise Exception(f'Incorrect number of arguments returned, filter: {params} result: {result}')

    def delete(self, pk, params: dict) -> BaseModel:
        resp = self.session.delete(url=f'{self.url}{pk}', params=params)
        self.process_response(resp)
        return self.model().deserialize(resp.json())

    def patch(self, pk, patch_data: dict, params: dict) -> BaseModel:
        resp = self.session.patch(url=f'{self.url}{pk}', params=params, data=patch_data)
        self.process_response(resp)
        return self.model().deserialize(resp.json())

    def create(self, create_data: dict, params: dict) -> BaseModel:
        resp = self.session.post(url=f'{self.url}', params=params, data=create_data)
        self.process_response(resp, expected_code=HTTP_201_CREATED)
        return self.model().deserialize(resp.json())
