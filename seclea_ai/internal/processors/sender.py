import logging
import os
import time
from typing import Dict, List

from .processor import Processor
from ..exceptions import BadRequestError
from ...internal.api.api_interface import Api
from ...internal.local_db import RecordStatus, Record

logger = logging.getLogger("seclea_ai")


# TODO wrap all db requests in transactions to reduce clashes.

"""
Exceptions to handle
- Database errors
"""


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
        # something went wrong - record in status and raise for handling in director.
        except Exception:
            tr_record.status = RecordStatus.SEND_FAIL.value
            tr_record.save()
            raise
        else:
            tr_record.status = RecordStatus.SENT.value
            tr_record.remote_id = response["id"]
            tr_record.save()
            return record_id

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
        # something went wrong - record in status and raise for handling in director.
        except Exception:
            record.status = RecordStatus.SEND_FAIL.value
            record.save()
            raise
        else:
            # update record status in sqlite - TODO refactor out to common function.
            record.remote_id = response["id"]  # TODO improve parsing.
            record.status = RecordStatus.SENT.value
            record.save()
            # clean up file
            os.remove(record.path)
            return record_id

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
            logger.debug(dataset_record.status)
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
        # something went wrong - record in status and raise for handling in director.
        except BadRequestError as e:
            # need to check content - if it's duplicate we need to get the remote id for use in other reqs
            logger.debug(e)
            if "already exists" in str(e):
                logger.debug(
                    f"Entity already exists, skipping DatasetTransformation, id: {record_id}"
                )
                dataset = self._api.get_dataset(
                    project_id=self._settings["project_id"],
                    organization_id=self._settings["organization"],
                    dataset_id=dataset_id,
                )
                dataset_record.remote_id = dataset["hash"]  # TODO make id (portal issue)
                dataset_record.status = RecordStatus.SENT.value
                dataset_record.save()
                os.remove(dataset_record.path)
                return record_id
        except Exception:
            dataset_record.status = RecordStatus.SEND_FAIL.value
            dataset_record.save()
            raise
        else:
            # update record status in sqlite
            dataset_record.remote_id = response[
                "hash"
            ]  # TODO improve parsing. - should be id - portal issue
            dataset_record.status = RecordStatus.SENT.value
            dataset_record.save()
            # clean up file
            os.remove(dataset_record.path)
            return record_id

    def _send_transformation(self, record_id, name, code_raw, code_encoded, **kwargs):
        record = Record.get_by_id(record_id)
        try:
            dataset_record_id = record.dependencies[0]
            dataset_record = Record.get_by_id(dataset_record_id)

            # wait for storage to complete if it hasn't
            start = time.time()
            give_up = 1.0
            while dataset_record.status != RecordStatus.SENT.value:
                logger.debug(dataset_record.status)
                if time.time() - start >= give_up:
                    raise TimeoutError("Waited too long for Dataset upload")
                time.sleep(0.1)
                dataset_record = Record.get_by_id(record_id)

            dataset_id = dataset_record.remote_id
        except IndexError:
            dataset_id = None

        try:
            response = self._api.upload_transformation(
                project_id=self._settings["project_id"],
                organization_id=self._settings["organization"],
                code_raw=code_raw,
                code_encoded=code_encoded,
                name=name,
                dataset_id=dataset_id,
            )
            # TODO improve/factor out validation and updating status - return error codes or something
        # something went wrong - record in status and raise for handling in director.
        except BadRequestError as e:
            # need to check content - if it's duplicate we need to get the remote id for use in other reqs
            logger.debug(e)
            if "already exists" in str(e):
                logger.debug(
                    f"Entity already exists, skipping DatasetTransformation, id: {record_id}"
                )
                transformations = self._api.get_transformations(
                    project_id=self._settings["project_id"],
                    organization_id=self._settings["organization"],
                    code_raw=code_raw,
                    name=name,
                    dataset_id=dataset_id,
                )
                record.remote_id = transformations[0]["id"]
                record.status = RecordStatus.SENT.value
                record.save()
                return
        except Exception:
            record.status = RecordStatus.SEND_FAIL.value
            record.save()
            raise
        else:
            record.status = RecordStatus.SENT.value
            record.remote_id = response["id"]
            record.save()
            return record_id
