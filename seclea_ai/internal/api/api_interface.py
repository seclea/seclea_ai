from requests import Session

from ...internal.authentication import AuthenticationService
from . import collection


class PlatformApi:
    """
    Api to converse with Seclea services provided credentials
    """

    def __init__(self, username, password, platform_url, auth_url):
        self.auth = AuthenticationService(url=auth_url)
        self.session = Session()
        self.auth.authenticate(self.session, username=username, password=password)

        collection_base_url = f"{platform_url}/collection/"
        self.organizations = collection.OrganizationApi(
            base_url=platform_url + "/", session=self.session
        )
        self.projects = collection.ProjectApi(base_url=collection_base_url, session=self.session)
        self.datasets = collection.DatasetApi(base_url=collection_base_url, session=self.session)
        self.dataset_transformations = collection.DatasetTransformationApi(
            base_url=collection_base_url, session=self.session
        )
        self.models = collection.ModelApi(base_url=collection_base_url, session=self.session)
        self.model_states = collection.ModelStateApi(
            base_url=collection_base_url, session=self.session
        )
