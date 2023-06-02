"""
Everything to do with the API to the backend.
"""
import json
import logging
import os
from typing import Dict, List, Union, Optional
from uuid import UUID

from circuitbreaker import circuit
from requests import Response, Session

from .api_utils import api_request, degraded_service_exceptions
from ...internal.authentication import AuthenticationService


class Api:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings, username=None, password=None):
        # setup some defaults
        self._settings = settings
        self._session = Session()
        self.auth = AuthenticationService(url=settings["auth_url"], session=self._session)
        # TODO maybe remove auth on creation - only when needed?
        self.auth.authenticate(username=username, password=password)
        self._root_url = settings["platform_url"]
        self._project_endpoint = "collection/projects"
        self._dataset_endpoint = "collection/datasets"
        self._dataset_transformations_endpoint = "collection/dataset-transformations"
        self._model_endpoint = "collection/models"
        self._training_run_endpoint = "collection/training-runs"
        self._model_states_endpoint = "collection/model-states"
        self._organization_endpoint = "organization"

    def __del__(self):
        self._session.close()

    def authenticate(self, username=None, password=None):
        self.auth.authenticate(username=username, password=password)

    @staticmethod
    def test_json_valid(d):
        d = json.dumps(d)
        json.loads(d)
        pass

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_organization_by_name(
        self, organization_name: str, **filter_kwargs
    ) -> Union[Response, List]:
        return self._session.get(
            url=f"{self._root_url}/{self._organization_endpoint}/",
            params={"name": organization_name, **filter_kwargs},
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_project(
        self, project_id: UUID, organization_id: UUID, **filter_kwargs
    ) -> Union[Response, Dict]:
        """

        :param project_id:
        :param organization_id:
        :param filter_kwargs:
        :return: Dict: The project as a dict (due to the api_request wrapper
        :raises: NotFoundError: If the project with the project_id is not found.
        """
        return self._session.get(
            url=f"{self._root_url}/{self._project_endpoint}/{str(project_id)}",
            params={"organization": str(organization_id), **filter_kwargs},
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_projects(self, organization_id: UUID, **filter_kwargs) -> Union[Response, List[Dict]]:
        return self._session.get(
            url=f"{self._root_url}/{self._project_endpoint}",
            params={"organization": str(organization_id), **filter_kwargs},
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def upload_project(
        self,
        organization_id: UUID,
        name: str,
        description: str,
    ) -> Union[Response, Dict]:
        return self._session.post(
            url=f"{self._root_url}/{self._project_endpoint}",
            json={
                "name": name,
                "description": description,
                "organization": str(organization_id),
            },
            params={"organization": str(organization_id)},
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_dataset(
        self, project_id: UUID, organization_id: UUID, dataset_id: str
    ) -> Union[Response, Dict]:
        return self._session.get(
            url=f"{self._root_url}/{self._dataset_endpoint}/{dataset_id}",
            params={"project": str(project_id), "organization": str(organization_id)},
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_datasets(
        self, project_id: UUID, organization_id: UUID, **filter_kwargs
    ) -> Union[Response, List[Dict]]:
        return self._session.get(
            url=f"{self._root_url}/{self._dataset_endpoint}",
            params={
                "project": str(project_id),
                "organization": str(organization_id),
                **filter_kwargs,
            },
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def upload_dataset(
        self,
        project_id: UUID,
        organization_id: UUID,
        dataset_file_path: str,
        name: str,
        metadata: dict,
        hash: int,
        parent_hash: Optional[int] = None,
        parent: Optional[UUID] = None,
        dataset_id: Optional[UUID] = None,
    ) -> Union[Response, Dict]:

        dataset_queryparams = {"project": str(project_id), "organization": str(organization_id)}
        self.test_json_valid(metadata)

        with open(dataset_file_path, "rb") as f:
            dataset_obj = {
                "uuid": (None, str(dataset_id) if dataset_id is not None else None),
                "project": (None, str(project_id)),
                "name": (None, name),
                "metadata": (None, json.dumps(metadata), "application/json"),
                "hash": (None, str(hash)),
                "dataset": (os.path.basename(dataset_file_path), f),
                "parent_hash": (None, str(parent_hash) if parent_hash is not None else None),
            }
            if parent is not None:
                dataset_obj["parent"] = (None, str(parent))
                logging.debug(f"dataset_obj dataset field: {dataset_obj['parent']}")

            return self._session.post(
                url=f"{self._root_url}/{self._dataset_endpoint}",
                files=dataset_obj,
                params=dataset_queryparams,
            )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_models(
        self, project_id: UUID, organization_id: UUID, **filter_kwargs
    ) -> Union[Response, List[Dict]]:
        """
        Get models - with optional filter parameters.

        :param project_id:

        :param organization_id:

        :param filter_kwargs: Optional filter parameters
            Available filter params are:
            - name: str The name of the model
            - framework: str The name of the framework of the model - see XXX for options

        :return: Response
        """
        return self._session.get(
            url=f"{self._root_url}/{self._model_endpoint}",
            params={
                "organization": str(organization_id),
                "project": str(project_id),
                **filter_kwargs,
            },
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def upload_model(
        self,
        project_id: UUID,
        organization_id: UUID,
        name: str,
        framework: str,
        model_id: Optional[UUID] = None,
    ) -> Union[Response, Dict]:
        return self._session.post(
            url=f"{self._root_url}/{self._model_endpoint}",
            json={
                "uuid": str(model_id) if model_id is not None else None,
                "organization": str(organization_id),
                "project": str(project_id),
                "name": name,
                "framework": framework,
            },
            params={"organization": str(organization_id), "project": str(project_id)},
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_training_runs(
        self, project_id: UUID, organization_id: UUID, **filter_kwargs
    ) -> Union[Response, List[Dict]]:
        return self._session.get(
            url=f"{self._root_url}/{self._training_run_endpoint}",
            params={
                "project": str(project_id),
                "organization": str(organization_id),
                **filter_kwargs,
            },
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def upload_training_run(
        self,
        project_id: UUID,
        organization_id: UUID,
        datasets: List[UUID],
        model: UUID,
        name: str,
        params: Dict,
        metadata: Dict,
        training_run_id: Optional[UUID] = None,
    ) -> Union[Response, Dict]:
        data = {
            "uuid": str(training_run_id) if training_run_id is not None else None,
            "organization": str(organization_id),
            "project": str(project_id),
            "datasets": [str(dataset_id) for dataset_id in datasets],
            "model": str(model),
            "name": name,
            "params": params,
            "metadata": metadata,
        }
        self.test_json_valid(data)

        return self._session.post(
            url=f"{self._root_url}/{self._training_run_endpoint}",
            json=data,
            params={"organization": str(organization_id), "project": str(project_id)},
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_model_states(
        self, project_id: UUID, organization_id: UUID, **filter_kwargs
    ) -> Union[Response, List[Dict]]:
        return self._session.get(
            url=f"{self._root_url}/{self._training_run_endpoint}",
            params={
                "project": str(project_id),
                "organization": str(organization_id),
                **filter_kwargs,
            },
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def upload_model_state(
        self,
        project_id: UUID,
        organization_id: UUID,
        model_state_file_path: str,
        training_run: UUID,
        sequence_num: int,
        model_state_id: Optional[UUID] = None,
    ) -> Union[Response, Dict]:
        with open(model_state_file_path, "rb") as f:
            return self._session.post(
                url=f"{self._root_url}/{self._model_states_endpoint}",
                files={
                    "uuid": (None, str(model_state_id) if model_state_id is not None else None),
                    "project": (None, str(project_id)),
                    "sequence_num": (None, sequence_num),
                    "training_run": (None, str(training_run)),
                    "state": (os.path.basename(model_state_file_path), f),
                },
                params={
                    "organization": str(organization_id),
                    "project": str(project_id),
                },
            )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def get_transformations(
        self, project_id: UUID, organization_id: UUID, **filter_kwargs
    ) -> Union[Response, List[Dict]]:
        return self._session.get(
            url=f"{self._root_url}/{self._dataset_transformations_endpoint}",
            params={
                "project": str(project_id),
                "organization": str(organization_id),
                **filter_kwargs,
            },
        )

    @circuit(expected_exception=degraded_service_exceptions)
    @api_request
    def upload_transformation(
        self,
        project_id: UUID,
        organization_id: UUID,
        name: str,
        code_raw,
        code_encoded,
        dataset: UUID,
        transformation_id: Optional[UUID] = None,
    ) -> Union[Response, Dict]:

        data = {
            "uuid": str(transformation_id) if transformation_id is not None else None,
            "name": name,
            "code_raw": code_raw,
            "code_encoded": code_encoded,
            "dataset": str(dataset),
        }
        return self._session.post(
            url=f"{self._root_url}/{self._dataset_transformations_endpoint}",
            json=data,
            params={"organization": str(organization_id), "project": str(project_id)},
        )
