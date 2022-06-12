import abc
import os
import queue
import time
import uuid
from abc import ABC
from multiprocessing import Queue
from typing import Dict, List

from pandas import DataFrame

from seclea_ai.internal.api import Api
from seclea_ai.internal.local_db import MyDatabase
from seclea_ai.lib.seclea_utils.core import CompressionFactory, save_object
from seclea_ai.lib.seclea_utils.model_management import ModelManagers, serialize


def _assemble_key(record) -> str:
    return f"{record['username']}-{record['project_id']}-{record['entity_id']}"


class Processor(ABC):

    _input_q: Queue
    # _result_q: Queue

    def __init__(self, settings, input_q: Queue, **kwargs):
        self._settings = settings
        self._input_q = input_q
        # self._result_q = result_q

    def __len__(self):
        return self._input_q.qsize()

    @abc.abstractmethod
    def handle(self, record):
        pass

    @abc.abstractmethod
    def terminate(self):
        pass


class Writer(Processor):

    _input_q: Queue
    # _result_q: Queue

    def __init__(self, settings, input_q: Queue):
        super().__init__(settings=settings, input_q=input_q)
        self._settings = settings
        self._input_q = input_q
        # self._result_q = result_q
        self._dbms = (
            MyDatabase()
        )  # TODO need a way to specify where db located - probably in .seclea

    def terminate(self) -> None:
        # need to finish processing all items in the queue otherwise we have data loss.
        while not self._input_q.empty():
            record = self._input_q.get(timeout=0.01)
            self.handle(record)

    def handle(self, record) -> None:
        # TODO check record type (if we have different ones) - maybe come back to this
        # self._write(record)
        funcs = {
            "dataset": self._save_dataset,
            "model_state": self._save_model_state,
        }
        entity = record["entity"]
        target_function = funcs[entity]
        del record["entity"]
        target_function(**record)

    def _write(self, record) -> None:
        """
        Writes a record to local storage. This is intended primarily as a buffer and backup if internet
        connectivity is not available.
        Record must contain:
            {
                username: str,
                project_id: int,
                entity_id: int, TODO something better that handles local uniqueness and preserves order for upload.
            }

        :param record:
        :return:
        """
        # key = _assemble_key(record)

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
            if not os.path.exists(self._settings["cache_dir"]):
                os.makedirs(self._settings["cache_dir"])

            dataset_path = os.path.join(self._settings["cache_dir"], f"{uuid.uuid4()}_tmp.csv")
            dataset.to_csv(dataset_path, index=True)
            comp_path = os.path.join(self._settings["cache_dir"], f"{uuid.uuid4()}_compressed")
            rb = open(dataset_path, "rb")
            comp_path = save_object(rb, comp_path, compression=CompressionFactory.ZSTD)
            # tidy up intermediate file
            os.remove(dataset_path)

            # update the record TODO refactor out.
            dataset_record = self._dbms.get_record(record_id=record_id)
            dataset_record.path = comp_path
            dataset_record.status = "stored"
            self._dbms.update_record(dataset_record)

        except Exception as e:
            # update the record TODO refactor out.
            dataset_record = self._dbms.get_record(record_id=record_id)
            dataset_record.status = "failed_to_store"
            self._dbms.update_record(record=dataset_record)
            print(e)

    def _save_model_state(
        self,
        record_id,
        model,
        training_run_pk: int,
        sequence_num: int,
        model_manager: ModelManagers,
        **kwargs,
    ):
        """
        Save model state in local temp directory
        """
        try:
            os.makedirs(
                os.path.join(
                    self._settings["cache_dir"],
                    f"{self._settings['project_name']}/{str(training_run_pk)}",
                ),
                exist_ok=True,
            )
            save_path = os.path.join(
                self._settings["cache_dir"],
                f"{self._settings['project_name']}/{training_run_pk}/model-{sequence_num}",
            )

            model_data = serialize(model, model_manager)
            save_path = save_object(model_data, save_path, compression=CompressionFactory.ZSTD)

            # update the record TODO refactor out.
            record = self._dbms.get_record(record_id=record_id)
            record.path = save_path
            record.status = "stored"
            self._dbms.update_record(record=record)

        except Exception as e:
            # update the record TODO refactor out.
            record = self._dbms.get_record(record_id=record_id)
            record.status = "failed_to_store"
            self._dbms.update_record(record=record)
            print(e)


class Sender(Processor):
    _input_q: Queue
    # _result_q: Queue
    _publish_q: Queue

    def __init__(self, settings, input_q: Queue):
        super().__init__(settings=settings, input_q=input_q)
        self._settings = settings
        self._input_q = input_q
        # self._result_q = result_q
        self._api = Api(settings=settings)
        self._dbms = MyDatabase()

    def handle(self, record) -> None:
        funcs = {
            "dataset": self._send_dataset,
            "model_state": self._send_model_state,
            "dataset_transformation": self._send_transformation,
            "training_run": self._send_training_run,
        }
        entity = record["entity"]
        target_function = funcs[entity]
        del record["entity"]
        target_function(**record)

    def terminate(self) -> None:
        # clear the queue otherwise will never join
        try:
            while True:
                self._input_q.get_nowait()
        except queue.Empty:
            print("exited")
            return

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
        response = self._api.upload_training_run(
            organization_pk=self._settings["organization"],
            project_pk=self._settings["project_id"],
            dataset_pks=dataset_pks,
            model_pk=model_pk,
            training_run_name=training_run_name,
            params=params,
        )

        # TODO improve/factor out validation and updating status - return error codes or something
        if response.status_code == 201:
            # update the record TODO refactor out.
            tr_record = self._dbms.get_record(record_id=record_id)
            tr_record.status = "uploaded"
            tr_record.remote_id = response.json()["id"]
            self._dbms.update_record(record=tr_record)
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the training run: {response.text()}' - {response.reason} - {response.text}"
            )

    def _send_model_state(self, record_id, sequence_num: int, final: bool, **kwargs):
        """
        Upload model state to server
        """
        # prep
        record = self._dbms.get_record(record_id)
        try:
            parent_record_id = record.dependencies[0]
            parent_record = self._dbms.get_record(parent_record_id)
            parent_id = parent_record.remote_id
        except IndexError:
            raise ValueError(
                "Training run must be uploaded before model state something went wrong"
            )

        # wait for storage to complete if it hasn't
        while record.status != "stored":
            time.sleep(0.5)
            record = self._dbms.get_record(
                record_id=record_id
            )  # TODO check if this is needed to update

        response = self._api.upload_model_state(
            model_state_file_path=record.path,
            organization_pk=self._settings["organization"],
            project_pk=self._settings["project_id"],
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
            self._dbms.update_record(record=record)
            os.remove(record.path)
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the dataset: {response.text()}' - {response.reason} - {response.text}"
            )

    def _send_dataset(
        self,
        record_id: int,
        metadata: Dict,
        dataset_name: str,
        dataset_hash,
        **kwargs,
    ):

        """
        Upload dataset file to server and upload transformation once dataset uploaded successfully
        """
        dataset_record = self._dbms.get_record(record_id)
        try:
            parent_record_id = dataset_record.dependencies[0]
            parent_record = self._dbms.get_record(record_id=parent_record_id)
            parent_id = parent_record.remote_id
        except IndexError:
            parent_id = None

        # wait for storage to complete if it hasn't
        while dataset_record.status != "stored":
            time.sleep(0.5)
            dataset_record = self._dbms.get_record(
                record_id=record_id
            )  # TODO check if this is needed to update

        response = self._api.upload_dataset(
            dataset_file_path=dataset_record.comp_path,
            project_pk=self._settings["project_id"],
            organization_pk=self._settings["organization"],
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
            self._dbms.update_record(dataset_record)
            os.remove(dataset_record.comp_path)
            # update the metadata - TODO remove and move to new uploading inside request.
            self._api.update_dataset_metadata(
                dataset_hash=dataset_hash,
                metadata=metadata,
                project=self._settings["project_id"],
                organization=self._settings["organization"],
            )
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the dataset: {response.text()}' - {response.reason} - {response.text}"
            )

    def _send_transformation(self, record_id, project, name, code_raw, code_encoded, **kwargs):
        record = self._dbms.get_record(record_id=record_id)
        try:
            dataset_record_id = record.dependencies[0]
            dataset_record = self._dbms.get_record(record_id=dataset_record_id)
            dataset_id = dataset_record.remote_id
        except IndexError:
            dataset_id = None

        response = self._api.upload_transformation(
            project=project,
            organization=self._settings["organization"],
            code_raw=code_raw,
            code_encoded=code_encoded,
            name=name,
            dataset_pk=dataset_id,
        )
        # TODO improve/factor out validation and updating status - return error codes or something
        if response.status_code == 201:
            # update the record TODO refactor out.
            record = self._dbms.get_record(record_id=record_id)
            record.status = "uploaded"
            record.remote_id = response.json()["id"]
            self._dbms.update_record(record=record)
        else:
            raise ValueError(
                f"Response Status code {response.status}, expected: 201. \n f'There was some issue uploading the training run: {response.text()}' - {response.reason} - {response.text}"
            )
