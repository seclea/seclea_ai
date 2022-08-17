import os
import time
import uuid
from abc import ABC
from pathlib import Path
from typing import Dict, List

from pandas import DataFrame
from peewee import SqliteDatabase

from seclea_ai.internal.api.api_interface import Api
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.lib.seclea_utils.object_management import Tracked


def _assemble_key(record) -> str:
    return f"{record['username']}-{record['project_id']}-{record['entity_id']}"


class Processor(ABC):
    def __init__(self, settings, **kwargs):
        self._settings = settings
        self._db = SqliteDatabase(
            Path.home() / ".seclea" / "seclea_ai.db",
            thread_safe=True,
            pragmas={"journal_mode": "wal"},
        )


# TODO wrap all db requests in transactions to reduce clashes.


class Writer(Processor):
    def __init__(self, settings):
        super().__init__(settings=settings)
        self._settings = settings
        self.funcs = {
            "dataset": self._save_dataset,
            "model_state": self._save_model_state,
        }

    def _save_dataset(
        self,
        record_id: int,
        dataset: DataFrame,
        **kwargs,  # TODO improve this interface to not need kwargs etc.
    ):
        """
        Save dataset in local temp directory
        """
        dataset_record = Record.get_by_id(record_id)
        try:
            # upload a dataset - only works for a single transformation.
            os.makedirs(self._settings["cache_dir"], exist_ok=True)

            # TODO take another look at this section.
            path_root = uuid.uuid4()
            dataset_path = self._settings["cache_dir"] / f"{path_root}_tmp.csv"
            dataset.to_csv(dataset_path, index=True)
            # with open(dataset_path, "rb") as rb:
            #     comp_path = save_object(
            #         rb,
            #         file_name=f"{path_root}_compressed",
            #         path=self._settings["cache_dir"],
            #         compression=CompressionFactory.ZSTD,
            #     )
            # tidy up intermediate file
            os.remove(dataset_path)

            # update the record TODO refactor out.
            # dataset_record.path = comp_path
            dataset_record.status = RecordStatus.STORED.value
            dataset_record.save()
            return record_id

        except Exception:
            # update the record TODO refactor out.
            dataset_record.status = RecordStatus.STORE_FAIL.value
            dataset_record.save()
            raise

    def _save_model_state(
        self,
        record_id,
        model: Tracked,
        sequence_num: int,
        **kwargs,
    ):
        """
        Save model state in local temp directory
        """
        record = Record.get_by_id(record_id)
        try:
            training_run_id = record.dependencies[0]
        except IndexError:
            raise ValueError(
                "Training run must be uploaded before model state something went wrong"
            )
        try:
            # TODO look again at this.
            save_path = self._settings["cache_dir"] / f"{str(training_run_id)}"
            os.makedirs(save_path, exist_ok=True)
            file_name = f"model-{sequence_num}"
            model.object_manager.full_path = save_path, file_name
            model.save_tracked()

            record.path = model.object_manager.full_path
            record.status = RecordStatus.STORED.value
            record.save()
            return record_id

        except Exception:
            record.status = RecordStatus.STORE_FAIL.value
            record.save()
            raise


class Sender(Processor):
    def __init__(self, settings, api: Api):
        super().__init__(settings=settings)
        self._settings = settings
        self._api = api
        self.funcs = {
            "dataset": self._send_dataset,
            "model_state": self._send_model_state,
            "transformation": self._send_transformation,
            "training_run": self._send_training_run,
        }

    def _send_training_run(
        self,
        record_id,
        project,
        training_run_name: str,
        model_id: int,
        dataset_ids: List[str],
        params: Dict,
        **kwargs,
    ):
        """
        :param training_run_name: eg. "Training Run 0"
        :param params: Dict The hyper parameters of the model - can auto extract?
        :return:
        """
        tr_record = Record.get_by_id(record_id)
        if project is None:
            raise Exception("You need to create a project before uploading a training run")
        try:
            response = self._api.upload_training_run(
                organization_id=self._settings["organization"],
                project_id=self._settings["project_id"],
                dataset_ids=dataset_ids,
                model_id=model_id,
                training_run_name=training_run_name,
                params=params,
            )
            tr_record.status = RecordStatus.SENT.value
            tr_record.remote_id = response.json()["id"]
            tr_record.save()
            return record_id
        # TODO improve error handling to requeue failures - also by different failure types
        except ValueError:
            tr_record.status = RecordStatus.SEND_FAIL.value
            tr_record.save()
            raise

    def _send_model_state(self, record_id, sequence_num: int, final: bool, **kwargs):
        """
        Upload model state to server
        """
        # prep
        record = Record.get_by_id(record_id)
        try:
            parent_record_id = record.dependencies[0]
            parent_record = Record.get_by_id(parent_record_id)
            parent_id = parent_record.remote_id
        except IndexError:
            raise ValueError(
                "Training run must be uploaded before model state something went wrong"
            )

        # wait for storage to complete if it hasn't
        start = time.time()
        give_up = 1.0
        while record.status != RecordStatus.STORED.value:
            if time.time() - start >= give_up:
                raise TimeoutError("Waited too long for Model State storage")
            time.sleep(0.1)
            record = Record.get_by_id(record_id)  # TODO check if this is needed to update

        try:
            response = self._api.upload_model_state(
                model_state_file_path=record.path,
                organization_id=self._settings["organization"],
                project_id=self._settings["project_id"],
                training_run_id=parent_id,
                sequence_num=sequence_num,
                final_state=final,
            )
            # update record status in sqlite - TODO refactor out to common function.
            record.remote_id = response.json()["id"]  # TODO improve parsing.
            record.status = RecordStatus.SENT.value
            record.save()
            # clean up file
            os.remove(record.path)
            return record_id
        except ValueError:
            record.status = RecordStatus.SEND_FAIL.value
            record.save()
            raise

    def _send_dataset(
        self,
        record_id: int,
        metadata: Dict,
        dataset_name: str,
        dataset_id,
        **kwargs,
    ):

        """
        Upload dataset file to server and upload transformation once dataset uploaded successfully
        """
        dataset_record = Record.get_by_id(record_id)
        try:
            parent_record_id = dataset_record.dependencies[0]
            parent_record = Record.get_by_id(parent_record_id)
            parent_id = parent_record.remote_id
        except TypeError:
            parent_id = None

        # wait for storage to complete if it hasn't
        start = time.time()
        give_up = 1.0
        while dataset_record.status != RecordStatus.STORED.value:
            print(dataset_record.status)
            if time.time() - start >= give_up:
                raise TimeoutError("Waited too long for Dataset Storage")
            time.sleep(0.1)
            dataset_record = Record.get_by_id(record_id)

        try:
            response = self._api.upload_dataset(
                dataset_file_path=dataset_record.path,
                project_id=self._settings["project_id"],
                organization_id=self._settings["organization"],
                name=dataset_name,
                metadata=metadata,
                dataset_id=dataset_id,
                parent_dataset_id=parent_id,
            )
            # update record status in sqlite
            dataset_record.remote_id = response.json()[
                "hash"
            ]  # TODO improve parsing. - should be id - portal issue
            dataset_record.status = RecordStatus.SENT.value
            dataset_record.save()
            # clean up file
            os.remove(dataset_record.path)
            return record_id
        except ValueError:
            dataset_record.status = RecordStatus.SEND_FAIL.value
            dataset_record.save()
            raise

    def _send_transformation(self, record_id, project, name, code_raw, code_encoded, **kwargs):
        record = Record.get_by_id(record_id)
        try:
            dataset_record_id = record.dependencies[0]
            dataset_record = Record.get_by_id(dataset_record_id)

            # wait for storage to complete if it hasn't
            start = time.time()
            give_up = 1.0
            while dataset_record.status != RecordStatus.SENT.value:
                print(dataset_record.status)
                if time.time() - start >= give_up:
                    raise TimeoutError("Waited too long for Dataset upload")
                time.sleep(0.1)
                dataset_record = Record.get_by_id(record_id)

            dataset_id = dataset_record.remote_id
        except IndexError:
            dataset_id = None

        try:
            response = self._api.upload_transformation(
                project_id=project,
                organization_id=self._settings["organization"],
                code_raw=code_raw,
                code_encoded=code_encoded,
                name=name,
                dataset_id=dataset_id,
            )
            # TODO improve/factor out validation and updating status - return error codes or something
            record.status = RecordStatus.SENT.value
            record.remote_id = response.json()["id"]
            record.save()
            return record_id
        except ValueError:
            record.status = RecordStatus.SEND_FAIL.value
            record.save()
            raise
