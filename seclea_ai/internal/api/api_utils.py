from functools import wraps
from typing import Callable, Any, Dict, List, Union

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


def handle_response(response: Response, msg: str = "") -> Response:
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


def api_request(func: Callable[..., Response]) -> Callable[..., Union[List, Dict]]:
    """
    Wraps a request to the api. It handles the response and unpacks the response to python types from json.
    :param func: The request function.
    :return: List | Dict depending on the request.
    :raises: any of the errors in handle_response.
    """

    @wraps(func)
    def inner(*args: Any, **kwargs: Any) -> Union[List, Dict]:
        return handle_response(func(*args, **kwargs)).json()

    return inner


def degraded_service_exceptions(exception_type, exception_value) -> bool:
    if exception_type is ServiceDegradedError:
        return True
    if exception_type is ServerError:
        return True
    return False
