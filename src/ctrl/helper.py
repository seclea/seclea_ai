# -*- coding: utf-8 -*-

__name__ = 'Helper'
__package__ = 'seclea_ai'

__doc__ = f""" introduction

    TODO: Write introduction for the helper
"""


#
# TODO: Need to write description for the function
#
def handle_response(res: Response, expected: int, msg: str) -> Response:
    """
    Handle responses from the server

    :param res: Response The response from the server.

    :param expected: int The expected HTTP status code (ie. 200, 201 etc.)

    :param msg: str The message to include in the Exception that is raised if
                the response doesn't have the expected status code

    :return: Response

    :raises: ValueError - if the response code doesn't match the expected
             code.

    """

    if not res.status_code == expected:
        raise ValueError(
            f"""Response Status code {res.status_code}, expected:{expected}.
            \n{msg} - {res.reason} - {res.text}"""
        )

    return res
