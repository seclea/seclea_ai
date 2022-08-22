from .status import HTTP_401_UNAUTHORIZED
from collections import defaultdict
from typing import Dict
from requests import Response


class ApiError(Exception):
    """
    Base api error to throw, when there is no set rule to handle the error code client-side
    """

    def __init__(self, resp: Response, msg=None):
        if msg is None:
            msg = f'Api error: {resp.status_code}, {resp.content}'
        super().__init__(msg)


class AuthenticationError(ApiError):
    """
    Exception for errors related to authentication
    """
    status_codes = [HTTP_401_UNAUTHORIZED]


API_ERROR_CODE_EXCEPTION_MAPPER: Dict[int,ApiError.__class__] = defaultdict(ApiError)

def _update_mapper(exception):
    API_ERROR_CODE_EXCEPTION_MAPPER.update(dict.fromkeys(exception.status_codes, exception))


_update_mapper(AuthenticationError)
