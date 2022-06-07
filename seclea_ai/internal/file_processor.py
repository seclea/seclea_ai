"""
File storing and uploading data to server
"""
import os
import tempfile
import uuid
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

    def _save_dataset(
        self,
        project: int,
        dataset: DataFrame,
        dataset_name: str,
        metadata: Dict,
        parent_hash: Union[int, None],
        transformation: Union[DatasetTransformation, None],
        onSuccess,
    ):
        """
        Save dataset in local temp directory and call functions to upload dataset
        """
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

            # store datasets info in sqlite db
            ds = DatasetModelstate(
                name=dataset_name,
                path=dataset_path,
                comp_path=comp_path,
                project=str(project),
                organization=self._organization,
            )

            self._dbms.save_datasetmodelstate(ds, "failed")
            onSuccess(
                project,
                metadata,
                parent_hash,
                transformation,
                dataset_name,
                dataset_hash,
                comp_path,
                dataset_path,
            )
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
        onSuccess,
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

            onSuccess(save_path, training_run_pk, sequence_num, final)
        except Exception as e:
            ds = DatasetModelstate(
                path=save_path,
                project=str(self._project),
                organization=self._organization,
                training_run=str(training_run_pk),
            )

            self._dbms.save_datasetmodelstate(ds, "failed")
            print(e)

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
