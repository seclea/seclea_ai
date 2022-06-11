"""
File storing and uploading data to server
"""
import asyncio
import os
import tempfile
import time
import uuid
from enum import Enum
from queue import Queue
from typing import Dict, List

from pandas import DataFrame

from seclea_ai.lib.seclea_utils.core import CompressionFactory, save_object
from seclea_ai.lib.seclea_utils.model_management.get_model_manager import ModelManagers, serialize

from .local_db import MyDatabase


class RecordStatus(Enum):
    IN_MEMORY = "in_memory"
    STORED = "stored"
    SENT = "sent"
    STORE_FAIL = "store_fail"
    SEND_FAIL = "send_fail"


class FileProcessor:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(
        self, project_name, organization, transmission, api, auth_service, storage_q, sender_q
    ):
        # setup some defaults
        self._transmission = transmission
        self._project = None
        self._project_name = project_name
        self._organization = organization
        self._cache_dir = os.path.join(tempfile.gettempdir(), ".seclea/cache")
        self._api = api
        self._auth_service = auth_service
        self._dbms = MyDatabase()
        self._storage_q = storage_q
        self._sender_q = sender_q

    # TODO pull these out - I think that the queues are not managed correctly here and the self references
    # will be a problem as not really referencing the same object as in different threads...
    def writer(self, _storage_q: Queue):
        funcs = {
            "dataset": self._save_dataset,
            "model_state": self._save_model_state,
        }
        self._storage_q = _storage_q
        while not self._storage_q.empty():
            print("writing")
            obj = self._storage_q.get()

            if obj:
                entity = obj["entity"]
                target_function = funcs[entity]
                del obj["entity"]
                target_function(**obj)
        print("exited")

    def sender(self, _sender_q: Queue = None):
        funcs = {
            "dataset": self._send_dataset,
            "model_state": self._send_model_state,
            "dataset_transformation": self._send_transformation,
            "training_run": self._send_training_run,
        }
        if _sender_q:
            self._sender_q = _sender_q
        while not self._sender_q.empty():
            print("sending")
            obj = self._sender_q.get()
            if obj:
                entity = obj["entity"]
                target_function = funcs[entity]
                del obj["entity"]
                target_function(**obj)
        print("exited")

    def record_dataset(self):
        """
        Needs
            project id
            organization id

            name
            metadata
            hash
            file
            parent hash - optional
        @return:
        """
        # create db object with id

        pass

    def record_dataset_transformation(self):
        """
        Needs
            DatasetTransformation

            project id
            organization id

            name
            code_raw
            code_encoded
            order
            dataset id/hash

        @return:
        """
        pass

    def record_training_run(self):
        """
        Needs
            project id
            organization id

            datasets
            model
            name
            params

        @return:
        """
        pass

    def record_model_state(self):
        """
        Needs
            project id
            organization id

            training_run id
            file
            sequence_num
            final

        @return:
        """
        pass

    def record_model(self):
        """
        Needs
            project id
            organization id

            name
            framework

        @return:
        """
        pass

    def _save_dataset(
        self,
        record_id: int,
        dataset: DataFrame,
        **kwargs,  # TODO improve this interface to not need kwargs etc.
    ):
        """
        Save dataset in local temp directory and call functions to upload dataset
        """
        try:
            # upload a dataset - only works for a single transformation.
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)

            dataset_path = os.path.join(self._cache_dir, f"{uuid.uuid4()}_tmp.csv")
            dataset.to_csv(dataset_path, index=True)
            comp_path = os.path.join(self._cache_dir, f"{uuid.uuid4()}_compressed")
            rb = open(dataset_path, "rb")
            comp_path = save_object(rb, comp_path, compression=CompressionFactory.ZSTD)

            # update the record TODO refactor out.
            dataset_record = self._dbms.session.get(record_id)
            dataset_record.comp_path = comp_path
            dataset_record.path = dataset_path  # TODO remove I think
            dataset_record.status = "stored"
            self._dbms.session.commit()

        except Exception as e:
            # update the record TODO refactor out.
            dataset_record = self._dbms.session.get(record_id)
            dataset_record.status = "failed_to_store"
            self._dbms.session.commit()
            print(e)

    def _send_dataset(
        self,
        record_id: int,
        project: int,
        metadata: Dict,
        dataset_name: str,
        dataset_hash,
        **kwargs,
    ):

        """
        Upload dataset file to server and upload transformation once dataset uploaded successfully
        """
        dataset_record = self._dbms.session.get(record_id)
        try:
            parent_record_id = self._dbms.get_record_dependencies(record_id)[0]
            parent_record = self._dbms.session.get(parent_record_id)
            parent_id = parent_record.remote_id
        except IndexError:
            parent_id = None

        # wait for storage to complete if it hasn't
        while dataset_record.status != "stored":
            time.sleep(0.5)
            dataset_record = self._dbms.session.get(
                record_id
            )  # TODO check if this is needed to update

        response = self._api.post_dataset(
            transmission=self._transmission,
            dataset_file_path=dataset_record.comp_path,
            project_pk=project,
            organization_pk=self._organization,
            name=dataset_name,
            metadata={},
            dataset_hash=str(dataset_hash),
            parent_dataset_hash=parent_id,
            delete=False,
        )

        # update the db
        if response.status == 201:
            # update record status in sqlite
            dataset_record.remote_id = response.json()["id"]  # TODO improve parsing.
            dataset_record.status = "uploaded"
            self._dbms.session.commit()
            self._api._update_dataset_metadata(
                dataset_hash=dataset_hash,
                metadata=metadata,
                transmission=self._transmission,
                project=project,
                organization=self._organization,
            )
            os.remove(dataset_record.comp_path)
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the dataset: {response.text()}' - {response.reason} - {response.text}"
            )

    def _send_transformation(self, record_id, transformation, project, **kwargs):
        try:
            dataset_record_id = self._dbms.get_record_dependencies(record_id)[0]
            dataset_record = self._dbms.session.get(dataset_record_id)
            dataset_id = dataset_record.remote_id
        except IndexError:
            dataset_id = None

        response = self._api._upload_transformation(
            transformation=transformation,
            dataset_pk=dataset_id,
            transmission=self._transmission,
            project=project,
            organization=self._organization,
        )
        # TODO improve/factor out validation and updating status - return error codes or something
        if response.status_code == 201:
            # update the record TODO refactor out.
            record = self._dbms.session.get(record_id)
            record.status = "uploaded"
            record.remote_id = response.json()["id"]
            self._dbms.session.commit()
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the training run: {response.text()}' - {response.reason} - {response.text}"
            )

    def _send_training_run(
        self,
        record_id,
        project,
        training_run_name: str,
        model_pk: int,
        dataset_pks: List[str],
        params: Dict,
        **kwargs,
    ):
        """
        :param training_run_name: eg. "Training Run 0"
        :param params: Dict The hyper parameters of the model - can auto extract?
        :return:
        """
        if project is None:
            raise Exception("You need to create a project before uploading a training run")
        response = asyncio.run(
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
        # TODO improve/factor out validation and updating status - return error codes or something
        if response.status_code == 201:
            # update the record TODO refactor out.
            tr_record = self._dbms.session.get(record_id)
            tr_record.status = "uploaded"
            tr_record.remote_id = response.json()["id"]
            self._dbms.session.commit()
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the training run: {response.text()}' - {response.reason} - {response.text}"
            )

    def _save_model_state(
        self,
        record_id,
        model,
        training_run_pk: int,
        sequence_num: int,
        final: bool,
        model_manager: ModelManagers,
        **kwargs,
    ):
        """
        Save model state in local temp directory
        """
        try:
            os.makedirs(
                os.path.join(self._cache_dir, f"{self._project_name}/{str(training_run_pk)}"),
                exist_ok=True,
            )
            save_path = os.path.join(
                self._cache_dir, f"{self._project_name}/{training_run_pk}/model-{sequence_num}"
            )

            model_data = serialize(model, model_manager)
            save_path = save_object(model_data, save_path, compression=CompressionFactory.ZSTD)

            # update the record TODO refactor out.
            record = self._dbms.session.get(record_id)
            record.path = save_path
            record.status = "stored"
            self._dbms.session.commit()

        except Exception as e:
            # update the record TODO refactor out.
            record = self._dbms.session.get(record_id)
            record.status = "failed_to_store"
            self._dbms.session.commit()
            print(e)

    def _send_model_state(
        self, record_id, training_run_pk: int, sequence_num: int, final: bool, **kwargs
    ):
        """
        Upload model state to server
        """
        # prep
        record = self._dbms.session.get(record_id)
        try:
            parent_record_id = self._dbms.get_record_dependencies(record_id)[0]
            parent_record = self._dbms.session.get(parent_record_id)
            parent_id = parent_record.remote_id
        except IndexError:
            raise ValueError(
                "Training run must be uploaded before model state something went wrong"
            )

        # wait for storage to complete if it hasn't
        while record.status != "stored":
            time.sleep(0.5)
            record = self._dbms.session.get(record_id)  # TODO check if this is needed to update

        response = self._api.post_model_state(
            transmission=self._transmission,
            model_state_file_path=record.path,
            organization_pk=self._organization,
            project_pk=self._project,
            training_run_pk=str(parent_id),
            sequence_num=sequence_num,
            final_state=final,
            delete=False,
        )

        # update the db
        if response.status == 201:
            # update record status in sqlite
            record.remote_id = response.json()["id"]  # TODO improve parsing.
            record.status = "uploaded"
            self._dbms.session.commit()
            os.remove(record.path)
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the dataset: {response.text()}' - {response.reason} - {response.text}"
            )
