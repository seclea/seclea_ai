class APIError(Exception):
    """
    Raised for API error responses that don't fit into the specific ones above
    """


class BadRequestError(APIError):
    """
    Raised for API errors
    400
    """

    pass


class AuthenticationError(APIError):
    """
    Raised for authentication errors - ie. the user is not authenticated (not known to server)
    401
    """

    pass


class AuthorizationError(APIError):
    """
    Raised for authorization errors - ie. user is known but doesn't have permission.
    403
    """

    pass


class NotFoundError(APIError):
    """
    Raised for resource not found errors
    404
    """

    pass


class RequestTimeoutError(APIError):
    """
    Raised on request timeout
    408
    """

    pass


class ImATeapotError(APIError):
    """
    Raised on I'm a teapot error code
    418
    """

    pass


class ServerError(APIError):
    """
    Raised for non service degradation server internal error responses
    501, 505 - 511
    """

    pass


class ServiceDegradedError(APIError):
    """
    Raised for server errors indicating service degradation.
    500, 502, 503, 504
    """

    pass


class StorageSpaceError(Exception):
    """
    Raised when available storage space would be exceeded.
    """

    pass
