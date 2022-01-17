import json
from types import SimpleNamespace
from seclea_ai.seclea_utils.core.transmission import Transmission
from ..errors import throws_api_err

root = '/collection'


@throws_api_err
def get_projects(transmission: Transmission) -> list:
    """ Gets list of projects from the collection app given api_root

    @param transmission: Transmission type used for communication with server
    @return: List of projects or empty list
    """
    return transmission.get(f'{root}/projects').json()