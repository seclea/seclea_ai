"""
File storing and uploading data to server
"""
import os
import tempfile
import uuid
from multiprocessing import Queue
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from pandas import DataFrame

from seclea_ai.lib.seclea_utils.core import CompressionFactory, save_object
from seclea_ai.lib.seclea_utils.model_management.get_model_manager import ModelManagers, serialize
from seclea_ai.transformations import DatasetTransformation

from .local_db import DatasetModelstate, MyDatabase, StatusMonitor


class FileProcessor:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, project_name, organization, transmission, api):
        # setup some defaults
        self._transmission = transmission
        self._project = None
        self._project_name = project_name
        self._organization = organization
        self._cache_dir = os.path.join(tempfile.gettempdir(), ".seclea/cache")
        self._api = api
        self._dbms = MyDatabase()
        self.dataset_q = Queue()
        self.mdoel_state_q = Queue()

    def _save_dataset(
        self,
        project: int,
        dataset: DataFrame,
        dataset_name: str,
        metadata: Dict,
        parent_hash: Union[int, None],
        transformation: Union[DatasetTransformation, None],
    ):
        """
        Save dataset in local temp directory and call functions to upload dataset
        """
        if parent_hash is not None:

            parent_hash = hash(parent_hash + self._project)
            # check parent exists - throw an error if not.
            res = self._transmission.get(
                url_path=f"/collection/datasets/{parent_hash}",
                query_params={"project": self._project, "organization": self._organization},
            )
            if not res.status_code == 200:
                raise AssertionError(
                    "Parent Dataset does not exist on the Platform. Please check your arguments and "
                    "that you have uploaded the parent dataset already"
                )
            parent_metadata = res.json()["metadata"]
            # deal with the splits - take the set one by default but inherit from parent if None

            if transformation.split is None:
                # check the parent split - inherit split
                metadata["split"] = parent_metadata["split"]
            try:
                if metadata["outcome_name"] is None:
                    pass
            except KeyError:
                try:
                    metadata["outcome_name"] = parent_metadata["outcome_name"]
                except KeyError:
                    metadata["outcome_name"] = None

        # ensure that required keys are present in metadata and have meaningful defaults.
        # outcome name is here in case there are no transformations. It still needs to be set.
        defaults_spec = dict(
            continuous_features=[],
            outcome_name=None,
            num_samples=len(dataset),
        )
        metadata = self._ensure_required_metadata(metadata=metadata, defaults_spec=defaults_spec)

        try:
            features = (
                dataset.columns
            )  # TODO - drop the outcome name but requires changes on frontend.
        except KeyError:
            # this means outcome was set to None
            features = dataset.columns

        required = dict(
            index=0 if dataset.index.name is None else dataset.index.name,
            split=transformation.split if transformation is not None else None,
            features=list(features),
        )
        metadata = self._add_required_metadata(metadata=metadata, required_spec=required)

        # constraints
        if not set(metadata["continuous_features"]).issubset(set(metadata["features"])):
            raise ValueError(
                "Continuous features must be a subset of features. Please check and try again."
            )
        try:
            self._project = project
            # upload a dataset - only works for a single transformation.
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)

            dataset_path = os.path.join(self._cache_dir, f"{uuid.uuid4()}_tmp.csv")
            dataset.to_csv(dataset_path, index=True)
            comp_path = os.path.join(self._cache_dir, f"{uuid.uuid4()}_compressed")
            rb = open(dataset_path, "rb")
            comp_path = save_object(rb, comp_path, compression=CompressionFactory.ZSTD)

            dataset_hash = hash(pd.util.hash_pandas_object(dataset).sum() + project)

            self.dataset_q.put(
                self.dataset_q.put(
                    {
                        "project": project,
                        "metadata": metadata,
                        "parent_hash": parent_hash,
                        "transformation": transformation,
                        "dataset_name": dataset_name,
                        "dataset_hash": dataset_hash,
                        "comp_path": comp_path,
                        "dataset_path": dataset_path,
                    }
                )
            )

            # store datasets info in sqlite db
            ds = DatasetModelstate(
                name=dataset_name,
                path=dataset_path,
                comp_path=comp_path,
                project=str(project),
                organization=self._organization,
            )

            self._dbms.save_datasetmodelstate(ds, "failed")

        except Exception as e:
            ds = DatasetModelstate(
                name=dataset_name,
                path=dataset_path,
                comp_path=comp_path,
                project=str(project),
                organization=self._organization,
            )
            self._dbms.save_datasetmodelstate(ds, "failed")
            print(e)

    def send_dataset(self):
        while not self.dataset_q.empty():
            obj = self.dataset_q.get()
            if obj:
                self._send_dataset(
                    project=obj["project"],
                    metadata=obj["metadata"],
                    parent_hash=obj["parent_hash"],
                    transformation=obj["transformation"],
                    dataset_name=obj["dataset_name"],
                    dataset_hash=obj["dataset_hash"],
                    comp_path=obj["comp_path"],
                    dataset_path=obj["dataset_path"],
                )

    def _send_dataset(
        self,
        project,
        metadata,
        parent_hash,
        transformation,
        dataset_name,
        dataset_hash,
        comp_path,
        dataset_path,
    ):

        """
        Upload dataset file to server and upload transformation once dataset uploaded successfully
        """
        response = self._api.post_dataset(
            transmission=self._transmission,
            dataset_file_path=comp_path,
            project_pk=project,
            organization_pk=self._organization,
            name=dataset_name,
            metadata={},
            dataset_hash=str(dataset_hash),
            parent_dataset_hash=str(parent_hash) if parent_hash is not None else None,
            delete=True,
        )

        # upload the transformations
        if response.status == 201:
            # update record status in sqlite
            obj = (
                self._dbms.session.query(DatasetModelstate)
                .filter(DatasetModelstate.path == dataset_path)
                .first()
            )
            status = (
                self._dbms.session.query(StatusMonitor).filter(StatusMonitor.pid == obj.id).first()
            )
            status.status = "uploaded"
            self._dbms.session.commit()
            self._api._update_dataset_metadata(
                dataset_hash=dataset_hash,
                metadata=metadata,
                transmission=self._transmission,
                project=project,
                organization=self._organization,
            )
            if transformation is not None:
                # upload transformations.
                self._api._upload_transformation(
                    transformation=transformation,
                    dataset_pk=str(dataset_hash),
                    transmission=self._transmission,
                    project=project,
                    organization=self._organization,
                )
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the dataset: {response.text()}' - {response.reason} - {response.text}"
            )

    def _save_model_state(
        self,
        model,
        training_run_pk: int,
        sequence_num: int,
        final: bool,
        model_manager: ModelManagers,
    ):
        """
        Save model state in local temp directory and call functions to upload model state
        """
        try:
            os.makedirs(
                os.path.join(self._cache_dir, str(training_run_pk)),
                exist_ok=True,
            )
            model_data = serialize(model, model_manager)
            save_path = os.path.join(
                Path.home(), f".seclea/{self._project_name}/{training_run_pk}/model-{sequence_num}"
            )

            save_path = save_object(model_data, save_path, compression=CompressionFactory.ZSTD)

            # store model state info in sqlite
            ds = DatasetModelstate(
                path=save_path,
                project=str(self._project),
                organization=self._organization,
                training_run=str(training_run_pk),
            )
            self._dbms.save_datasetmodelstate(ds, "stored")
            # pushing data to queue
            self.mdoel_state_q.put(
                {
                    "save_path": save_path,
                    "training_run_pk": training_run_pk,
                    "sequence_num": sequence_num,
                    "final": final,
                }
            )
        except Exception as e:
            ds = DatasetModelstate(
                path=save_path,
                project=str(self._project),
                organization=self._organization,
                training_run=str(training_run_pk),
            )

            self._dbms.save_datasetmodelstate(ds, "failed")
            print(e)

    def send_model_state(self):
        while not self.mdoel_state_q.empty():
            obj = self.mdoel_state_q.get()
            if obj:
                self._send_model_state(
                    save_path=obj["save_path"],
                    training_run_pk=obj["training_run_pk"],
                    sequence_num=obj["sequence_num"],
                    final=obj["final"],
                )

    def _send_model_state(self, save_path, training_run_pk: int, sequence_num: int, final: bool):
        """
        Upload model state to server
        """
        res = self._api.post_model_state(
            self._transmission,
            save_path,
            self._organization,
            self._project,
            str(training_run_pk),
            sequence_num,
            final,
            True,
        )

        if res.status == 201:
            # update record status in sqlite
            obj = (
                self._dbms.session.query(DatasetModelstate)
                .filter(DatasetModelstate.path == save_path)
                .first()
            )
            status = (
                self._dbms.session.query(StatusMonitor).filter(StatusMonitor.pid == obj.id).first()
            )
            status.status = "uploaded"
            self._dbms.session.commit()
        else:
            raise ValueError(
                f"Response Status code {res.status}, expected: 201. \n f'There was some issue uploading a model state: {res.text()}' - {res.reason} - {res.text}"
            )
        return res

    @staticmethod
    def _ensure_required_metadata(metadata: Dict, defaults_spec: Dict) -> Dict:
        """
        Ensures that required metadata that can be specified by the user are filled.
        @param metadata: The metadata dict
        @param defaults_spec:
        @return: metadata
        """
        for required_key, default in defaults_spec.items():
            try:
                if metadata[required_key] is None:
                    metadata[required_key] = default
            except KeyError:
                metadata[required_key] = default
        return metadata

    @staticmethod
    def _add_required_metadata(metadata: Dict, required_spec: Dict) -> Dict:
        """
        Adds required - non user specified fields to the metadata
        @param metadata: The metadata dict
        @param required_spec:
        @return: metadata
        """
        for required_key, default in required_spec.items():
            metadata[required_key] = default
        return metadata
