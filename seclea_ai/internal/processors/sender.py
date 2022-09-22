import logging
import os
import time
from typing import Dict

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
        training_run_name: str,
        model_id: str,
        params: Dict,
        **kwargs,
    ):
        """
        :param training_run_name: eg. "Training Run 0"
        :param params: Dict The hyper parameters of the model - can auto extract?
        :return:
        """
        tr_record = Record.get_by_id(record_id)
        dataset_ids = list()

        # get the remote id's of the datasets from the db (they must be there due to send ordering)
        for dependency in tr_record.dependencies:
            record = Record.get_by_id(dependency)
            dataset_ids.append(record.remote_id)

        try:
            response = self._api.upload_training_run(
                organization_id=self._settings["organization_id"],
                project_id=self._settings["project_id"],
                dataset_ids=dataset_ids,
                model_id=model_id,
                training_run_name=training_run_name,
                params=params,
            )
        except BadRequestError as e:
            # need to check content - if it's duplicate we need to get the remote id for use in other reqs
            logger.debug(e)
            if "already exists" in str(e):
                logger.warning(f"Entity already exists, skipping TrainingRun, id: {record_id}")
                training_runs = self._api.get_training_runs(
                    organization_id=self._settings["organization_id"],
                    project_id=self._settings["project_id"],
                    name=training_run_name,
                    dataset_ids=dataset_ids,
                    model_id=model_id,
                )
                tr_record.remote_id = training_runs[0]["uuid"]
                tr_record.status = RecordStatus.SENT.value
                tr_record.save()
                return record_id
            else:
                tr_record.status = RecordStatus.SEND_FAIL.value
                tr_record.save()
                raise
        # something went wrong - record in status and raise for handling in director.
        except Exception:
            tr_record.status = RecordStatus.SEND_FAIL.value
            tr_record.save()
            raise
        else:
            tr_record.status = RecordStatus.SENT.value
            tr_record.remote_id = response["uuid"]
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
                organization_id=self._settings["organization_id"],
                project_id=self._settings["project_id"],
                training_run_id=parent_id,
                sequence_num=sequence_num,
                final_state=final,
            )
        except BadRequestError as e:
            # need to check content - if it's duplicate we need to get the remote id for use in other reqs
            logger.debug(e)
            if "already exists" in str(e):
                logger.warning(f"Entity already exists, skipping ModelState, id: {record_id}")
                model_state = self._api.get_model_states(
                    project_id=self._settings["project_id"],
                    organization_id=self._settings["organization_id"],
                    training_run_id=parent_id,
                    sequence_num=sequence_num,
                )
                record.remote_id = model_state[0]["uuid"]
                record.status = RecordStatus.SENT.value
                record.save()
                os.remove(record.path)
                return record_id
            else:
                record.status = RecordStatus.SEND_FAIL.value
                record.save()
                raise
        # something went wrong - record in status and raise for handling in director.
        except Exception:
            record.status = RecordStatus.SEND_FAIL.value
            record.save()
            raise
        else:
            # update record status in sqlite - TODO refactor out to common function.
            record.remote_id = response["uuid"]  # TODO improve parsing.
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
        dataset_hash: int,
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
                organization_id=self._settings["organization_id"],
                name=dataset_name,
                metadata=metadata,
                dataset_hash=dataset_hash,
                parent_dataset_id=parent_id,
            )
        # something went wrong - record in status and raise for handling in director.
        except BadRequestError as e:
            # need to check content - if it's duplicate we need to get the remote id for use in other reqs
            logger.debug(e)
            if "already exists" in str(e):
                logger.warning(f"Entity already exists, skipping Dataset, id: {record_id}")
                dataset = self._api.get_datasets(
                    project_id=self._settings["project_id"],
                    organization_id=self._settings["organization_id"],
                    hash=dataset_hash,
                )
                dataset_record.remote_id = dataset[0]["uuid"]
                dataset_record.status = RecordStatus.SENT.value
                dataset_record.save()
                os.remove(dataset_record.path)
                return record_id
            else:
                dataset_record.status = RecordStatus.SEND_FAIL.value
                dataset_record.save()
                raise
        except Exception:
            dataset_record.status = RecordStatus.SEND_FAIL.value
            dataset_record.save()
            raise
        else:
            # update record status in sqlite
            dataset_record.remote_id = response["uuid"]  # TODO improve parsing.
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
                organization_id=self._settings["organization_id"],
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
                logger.warning(
                    f"Entity already exists, skipping DatasetTransformation, id: {record_id}"
                )
                transformations = self._api.get_transformations(
                    project_id=self._settings["project_id"],
                    organization_id=self._settings["organization_id"],
                    code_raw=code_raw,
                    name=name,
                    dataset_id=dataset_id,
                )
                record.remote_id = transformations[0]["uuid"]
                record.status = RecordStatus.SENT.value
                record.save()
                return
            else:
                record.status = RecordStatus.SEND_FAIL.value
                record.save()
                raise
        except Exception:
            record.status = RecordStatus.SEND_FAIL.value
            record.save()
            raise
        else:
            record.status = RecordStatus.SENT.value
            record.remote_id = response["uuid"]
            record.save()
            return record_id
