class AuthenticationError(Exception):
    """
    Raised for any authentication error.
    """

    pass


class APIError(Exception):
    """
    Raised for any API error. TODO decide if this helps or not. Maybe have subclasses for specific errors.
    """

    pass
