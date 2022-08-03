class BadRequestError(Exception):
    """
    Raised for API errors
    400
    """

    pass


class AuthenticationError(Exception):
    """
    Raised for authentication errors - ie. the user is not authenticated (not known to server)
    401
    """

    pass


class AuthorizationError(Exception):
    """
    Raised for authorization errors - ie. user is known but doesn't have permission.
    403
    """

    pass


class NotFoundError(Exception):
    """
    Raised for resource not found errors - 404.
    """

    pass


class ServerError(Exception):
    """
    Raised for server error responses (500 codes)
    500 - 511 - maybe split some out later
    """

    pass


class APIError(Exception):
    """
    Raised for API error responses that don't fit into the specific ones above
    """
