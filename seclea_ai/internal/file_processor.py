"""
File storing and uploading data to server
"""
import asyncio
import os
import tempfile
import uuid
from queue import Queue
from typing import Dict, List, Union

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

    def __init__(self, project_name, organization, transmission, api, auth_service):
        # setup some defaults
        self._transmission = transmission
        self._project = None
        self._project_name = project_name
        self._organization = organization
        self._cache_dir = os.path.join(tempfile.gettempdir(), ".seclea/cache")
        self._api = api
        self._auth_service = auth_service
        self._dbms = MyDatabase()
        self._sender_q = Queue()

    def writer(self, _storage_q: Queue):
        self._storage_q = _storage_q
        while not self._storage_q.empty():
            obj = self._storage_q.get()

            if obj:
                if obj["function"] == "upload_dataset":
                    self._save_dataset(
                        obj["project"],
                        obj["dataset"],
                        obj["dataset_name"],
                        obj["metadata"],
                        obj["parent_hash"],
                        obj["transformation"],
                    )
                if obj["function"] == "upload_training_run":
                    self.upload_training_run(
                        model=obj["model"],
                        train_dataset=obj["train_dataset"],
                        test_dataset=obj["test_dataset"],
                        val_dataset=["val_dataset"],
                        project=obj["project"],
                    )
                if obj["function"] == "_save_model_state":
                    self._save_model_state(
                        model=obj["model"],
                        training_run_pk=obj["training_run_pk"],
                        sequence_num=obj["sequence_num"],
                        final=obj["final"],
                        model_manager=obj["model_manager"],
                    )

    def sender(self):
        while not self._sender_q.empty():
            obj = self._sender_q.get()
            if obj:
                if obj["function"] == "_send_model_state":
                    self._send_model_state(
                        save_path=obj["save_path"],
                        training_run_pk=obj["training_run_pk"],
                        sequence_num=obj["sequence_num"],
                        final=obj["final"],
                    )
                if obj["function"] == "_send_dataset":
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
                return

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

            # store datasets info in sqlite db
            ds = DatasetModelstate(
                name=dataset_name,
                path=dataset_path,
                comp_path=comp_path,
                project=str(project),
                organization=self._organization,
            )

            self._dbms.save_datasetmodelstate(ds, "stored")

            # self._send_dataset(project, metadata, parent_hash, transformation, dataset_name, dataset_hash, comp_path,
            #                   dataset_path)
            self._sender_q.put(
                {
                    "function": "_send_dataset",
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

    def upload_training_run(
        self,
        model,
        project,
        train_dataset: DataFrame,
        test_dataset: DataFrame = None,
        val_dataset: DataFrame = None,
    ) -> None:
        """
        Takes a model and extracts the necessary data for uploading the training run.

        :param model: An ML Model instance. This should be one of {sklearn.Estimator, xgboost.Booster, lgbm.Boster}.

        :param train_dataset: DataFrame The Dataset that the model is trained on.

        :param test_dataset: DataFrame The Dataset that the model is trained on.

        :param val_dataset: DataFrame The Dataset that the model is trained on.

        :return: None

        Example::

            >>> seclea = SecleaAI(project_name="Test Project")
            >>> dataset = pd.read_csv(<dataset_name>)
            >>> model = LogisticRegressionClassifier()
            >>> model.fit(X, y)
            >>> seclea.upload_training_run(
                    model,
                    framework=seclea_ai.Frameworks.SKLEARN,
                    dataset_name="Test Dataset",
                )
        """
        self._auth_service.authenticate(self._transmission)

        # validate the splits? maybe later when we have proper Dataset class to manage these things.
        dataset_pks = [
            str(hash(pd.util.hash_pandas_object(dataset).sum() + project))
            for dataset in [train_dataset, test_dataset, val_dataset]
            if dataset is not None
        ]

        model_name = model.__class__.__name__

        framework = self._get_framework(model)

        # check the model exists upload if not
        model_type_pk = self._set_model(model_name=model_name, framework=framework)

        # check the latest training run
        training_runs_res = self._transmission.get(
            "/collection/training-runs",
            query_params={
                "project": self._project,
                "model": model_type_pk,
                "organization": self._organization,
            },
        )
        training_runs = training_runs_res.json()

        # Create the training run name
        largest = -1
        for training_run in training_runs:
            num = int(training_run["name"].split(" ")[2])
            if num > largest:
                largest = num
        training_run_name = f"Training Run {largest + 1}"

        # extract params from the model
        params = framework.value.get_params(model)

        self._storage_q.put(
            {
                "function": "_upload_training_run",
                "project": project,
                "mdoel": model,
                "framework": framework,
                "training_run_name": training_run_name,
                "model_pk": model_type_pk,
                "dataset_pks": dataset_pks,
                "params": params,
            }
        )

    def _upload_training_run(
        self,
        project,
        model,
        framework,
        training_run_name: str,
        model_pk: int,
        dataset_pks: List[str],
        params: Dict,
    ):
        """

        :param training_run_name: eg. "Training Run 0"
        :param params: Dict The hyper parameters of the model - can auto extract?
        :return:
        """
        if project is None:
            raise Exception("You need to create a project before uploading a training run")
        res = asyncio.run(
            self._api.send_json(
                url_path="/collection/training-runs",
                obj={
                    "organization": self._organization,
                    "project": self._project,
                    "datasets": dataset_pks,
                    "model": model_pk,
                    "name": training_run_name,
                    "params": params,
                },
                query_params={"organization": self._organization, "project": self._project},
                transmission=self._transmission,
                json_response=True,
            )
        )
        # if the upload was successful, add the new training_run to the list to keep the names updated.
        self._training_run = res["id"]
        self._storage_q.put(
            {
                "function": "_save_model_state",
                "model": model,
                "training_run_pk": self._training_run,
                "sequence_num": 0,
                "final": True,
                "model_manager": framework,
            }
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
                os.path.join(self._cache_dir, f"{self._project_name}/{str(training_run_pk)}"),
                exist_ok=True,
            )
            model_data = serialize(model, model_manager)
            save_path = os.path.join(
                self._cache_dir, f"{self._project_name}/{training_run_pk}/model-{sequence_num}"
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
            self._sender_q.put(
                {
                    "function": "_send_model_state",
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
