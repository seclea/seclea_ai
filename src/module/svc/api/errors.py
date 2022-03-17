class APIError(Exception):
    pass


def throws_api_err(f):
    def call(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise APIError(e)

    return call
