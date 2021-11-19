"""
Everything to do with the API to the backend.
"""
from seclea_utils.core.transmission import RequestWrapper

from seclea_ai.authentication import AuthenticationService


class Api:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings):
        # setup some defaults
        self._settings = settings
        self.transport = RequestWrapper(server_root_url=self._settings["platform_url"])
        self.auth = AuthenticationService.get_instance()
        self.auth.authenticate(self.transport)
        self.project_endpoint = "/collection/projects"
        self.dataset_endpoint = "/collection/datasets"
        self.model_endpoint = "/collection/models"
        self.training_run_endpoint = "/collection/training-runs"
        self.model_states_endpoint = "/collection/model-states"

    def reauthenticate(self):
        if not self.auth.verify_token(transmission=self.transport) and not self.auth.refresh_token(
            transmission=self.transport
        ):
            self.auth.authenticate(transmission=self.transport)
