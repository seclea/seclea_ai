import logging
import os
import time
from typing import Dict

from .processor import Processor
from ..exceptions import BadRequestError
from ..persistence.models import (
    Dataset,
    DatasetTransformation,
    ModelState,
    TrainingRun,
    TrainingRunDataset,
)
from ..schemas import (
    DatasetSchema,
    DatasetTransformationSchema,
    ModelStateSchema,
    TrainingRunSchema,
)
from ...internal.api.api_interface import Api
from ...internal.persistence.record import RecordStatus

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
        self.func_map = {
            DatasetSchema: self._send_dataset,
            DatasetTransformationSchema: self._send_transformation,
            TrainingRunSchema: self._send_training_run,
            ModelStateSchema: self._send_model_state,
        }

    def _send_training_run(
        self,
        training_run_schema,
        **kwargs,
    ):
        """
        :param training_run_name: eg. "Training Run 0"
        :param params: Dict The hyperparameters of the model - can auto extract?
        :return:
        """
        training_run_schema = TrainingRunSchema.parse_obj(training_run_schema)
        training_run_entity = TrainingRun.get(TrainingRun.uuid == training_run_schema.uuid)

        # get dataset uuids from many to many field.
        dataset_ids = [
            dataset.uuid
            for dataset in Dataset.select()
            .join(TrainingRunDataset)
            .join(TrainingRun)
            .where(TrainingRun.uuid == training_run_entity.uuid)
        ]

        try:
            self._api.upload_training_run(
                training_run_id=training_run_entity.uuid,
                organization_id=self._settings["organization_id"],
                project_id=self._settings["project_id"],
                dataset_ids=dataset_ids,
                model_id=training_run_entity.model.uuid,
                training_run_name=training_run_entity.name,
                params=training_run_entity.params,
            )
        except BadRequestError as e:
            # need to check content - if it's duplicate we need to get the remote id for use in other reqs
            logger.debug(e)
            if "already exists" in str(e):
                logger.warning(
                    f"Entity already exists, skipping TrainingRun, "
                    f"id: {training_run_entity.uuid}"
                )
                training_runs = self._api.get_training_runs(
                    organization_id=self._settings["organization_id"],
                    project_id=training_run_entity.project.uuid,
                    name=training_run_entity.name,
                    dataset_ids=dataset_ids,
                    model_id=training_run_entity.model.uuid,
                )
                training_run_entity.uuid = training_runs[0]["uuid"]
                training_run_entity.dataset = None
                training_run_entity.record.status = RecordStatus.SENT
                training_run_entity.record.save()
                training_run_entity.save()
                return training_run_entity.uuid
            else:
                training_run_entity.record.status = RecordStatus.SEND_FAIL
                training_run_entity.record.save()
                training_run_entity.save()
                raise
        # something went wrong - record in status and raise for handling in director.
        except Exception:
            training_run_entity.record.status = RecordStatus.SEND_FAIL
            training_run_entity.record.save()
            training_run_entity.save()
            raise
        else:
            training_run_entity.record.status = RecordStatus.SENT
            training_run_entity.record.save()
            training_run_entity.save()
            return training_run_entity.uuid

    def _send_model_state(self, model_state_schema: Dict, **kwargs):
        """
        Upload model state to server
        """
        # validate input data
        model_state_schema = ModelStateSchema.parse_obj(model_state_schema)
        model_state_entity = ModelState.get(ModelState.uuid == model_state_schema.uuid)

        # wait for storage to complete if it hasn't
        start = time.time()
        give_up = 3.0
        while model_state_entity.record.status != RecordStatus.STORED:
            if time.time() - start >= give_up:
                raise TimeoutError("Waited too long for Model State storage")
            time.sleep(0.1)
            # TODO check if this is needed to update
            model_state_entity = ModelState.get(ModelState.uuid == model_state_schema.uuid)

        try:
            self._api.upload_model_state(
                model_state_id=model_state_entity.uuid,
                model_state_file_path=model_state_entity.state,
                organization_id=self._settings["organization_id"],
                project_id=model_state_entity.training_run.project.uuid,
                training_run_id=model_state_entity.training_run.uuid,
                sequence_num=model_state_entity.sequence_num,
            )
        except BadRequestError as e:
            # need to check content - if it's duplicate we need to get the remote id for use in other reqs
            logger.debug(e)
            if "already exists" in str(e):
                logger.warning(
                    f"Entity already exists, skipping ModelState, " f"id: {model_state_entity.uuid}"
                )
                model_state = self._api.get_model_states(
                    project_id=model_state_entity.training_run.project,
                    organization_id=self._settings["organization_id"],
                    training_run_id=model_state_entity.training_run.uuid,
                    sequence_num=model_state_entity.sequence_num,
                )
                model_state_entity.uuid = model_state[0]["uuid"]
                model_state_entity.record.status = RecordStatus.SENT
                model_state_entity.record.save()
                model_state_entity.save()
                os.remove(model_state_entity.state)
                return model_state_entity.uuid
            else:
                model_state_entity.record.status = RecordStatus.SEND_FAIL
                model_state_entity.record.save()
                model_state_entity.save()
                raise
        # something went wrong - record in status and raise for handling in director.
        except Exception:
            model_state_entity.record.status = RecordStatus.SEND_FAIL
            model_state_entity.record.save()
            model_state_entity.save()
            raise
        else:
            # update record status in sqlite - TODO refactor out to common function.
            model_state_entity.record.status = RecordStatus.SENT
            model_state_entity.record.save()
            model_state_entity.save()
            # clean up file
            os.remove(model_state_entity.state)
            return model_state_entity

    def _send_dataset(
        self,
        dataset_schema: Dict,
        **kwargs,
    ):

        """
        Upload dataset file to server and upload transformation once dataset uploaded successfully
        """
        # validate input data
        dataset_schema = DatasetSchema.parse_obj(dataset_schema)
        dataset_entity = Dataset.get(Dataset.uuid == dataset_schema.uuid)
        try:
            parent_id = dataset_entity.parent.uuid
        except AttributeError:
            parent_id = None

        # wait for storage to complete if it hasn't
        start = time.time()
        give_up = 3.0
        while dataset_entity.record.status != RecordStatus.STORED:
            logger.debug(dataset_entity.record.status)
            if time.time() - start >= give_up:
                raise TimeoutError("Waited too long for Dataset Storage")
            time.sleep(0.1)
            dataset_entity = Dataset.get_by_id(dataset_entity.id)

        try:
            self._api.upload_dataset(
                dataset_id=dataset_entity.uuid,
                dataset_file_path=dataset_entity.dataset,
                project_id=dataset_entity.project.uuid,
                organization_id=self._settings["organization_id"],  # TODO check
                name=dataset_entity.name,
                metadata=dataset_entity.metadata,
                dataset_hash=dataset_entity.hash,
                parent_dataset_id=parent_id,
            )
        # something went wrong - record in status and raise for handling in director.
        except BadRequestError as e:
            # TODO check this - need to see if generating uuid before sending is an
            #  issue. The whole flow may need a rethink. Updating with the remote
            #  dataset info?
            # need to check content - if it's duplicate we need to get the remote id
            # for use in other reqs
            logger.debug(e)
            if "already exists" in str(e) or "must make a unique set." in str(e):
                logger.warning(
                    f"Entity already exists, skipping Dataset, " f"id: {dataset_entity.id}"
                )
                dataset_list = self._api.get_datasets(
                    project_id=dataset_entity.project.uuid,
                    organization_id=self._settings["organization_id"],
                    hash=dataset_entity.hash,
                )
                dataset_entity.record.status = RecordStatus.SENT
                dataset_entity.record.save()
                remote_dataset = DatasetSchema.parse_obj(dataset_list[0])
                dataset_entity.uuid = remote_dataset.uuid
                dataset_entity.save()
                os.remove(dataset_entity.dataset)
                return dataset_entity.uuid
            else:
                dataset_entity.record.status = RecordStatus.SEND_FAIL
                dataset_entity.record.save()
                dataset_entity.save()
                raise
        except Exception:
            dataset_entity.record.status = RecordStatus.SEND_FAIL
            dataset_entity.record.save()
            dataset_entity.save()
            raise
        else:
            # update record status in sqlite
            dataset_entity.record.status = RecordStatus.SENT
            dataset_entity.record.save()
            dataset_entity.save()
            # clean up file
            os.remove(dataset_entity.dataset)
            return dataset_entity.uuid

    def _send_transformation(self, transformation_schema: Dict, **kwargs):

        # validate input data
        transformation_schema = DatasetTransformationSchema.parse_obj(transformation_schema)
        transformation_entity = DatasetTransformation.get(
            DatasetTransformation.uuid == transformation_schema.uuid
        )

        # check the Dataset sent - TODO probably remove now that all uploads are
        #  sequential?

        dataset = Dataset.get_by_id(transformation_entity.dataset.id)

        # wait for storage to complete if it hasn't
        start = time.time()
        give_up = 1.0
        while dataset.record.status != RecordStatus.SENT:
            logger.debug(dataset.record.status)
            if time.time() - start >= give_up:
                raise TimeoutError("Waited too long for Dataset upload")
            time.sleep(0.1)
            dataset = Dataset.get_by_id(transformation_entity.dataset.id)

        dataset_id = dataset.uuid

        try:
            self._api.upload_transformation(
                transformation_id=transformation_schema.uuid,
                project_id=transformation_schema.dataset.project.uuid,
                organization_id=self._settings["organization_id"],
                code_raw=transformation_schema.code_raw,
                code_encoded=transformation_schema.code_encoded,
                name=transformation_schema.name,
                dataset_id=transformation_schema.dataset.uuid,
            )
            # TODO improve/factor out validation and updating status - return error
            #  codes or something
        # something went wrong - record in status and raise for handling in director.
        except BadRequestError as e:
            # need to check content - if it's duplicate we need to get the remote id
            # for use in other reqs
            logger.debug(e)
            if "already exists" in str(e):
                logger.warning(
                    f"Entity already exists, skipping DatasetTransformation, "
                    f"id: {transformation_schema.uuid}"
                )
                transformations = self._api.get_transformations(
                    project_id=transformation_schema.dataset.project.uuid,
                    organization_id=self._settings["organization_id"],
                    code_raw=transformation_schema.code_raw,
                    code_encoded=transformation_schema.code_encoded,
                    name=transformation_schema.name,
                    dataset_id=dataset_id,
                )
                # one option for validation and updating from remote
                # TODO make all consistent
                remote_transformation = DatasetTransformationSchema.parse_obj(transformations[0])
                transformation_entity.uuid = remote_transformation.uuid
                transformation_entity.record.status = RecordStatus.SENT
                transformation_entity.record.save()
                transformation_entity.save()
                return
            else:
                transformation_entity.record.status = RecordStatus.SEND_FAIL
                transformation_entity.record.save()
                transformation_entity.save()
                raise
        except Exception:
            transformation_entity.record.status = RecordStatus.SEND_FAIL
            transformation_entity.record.save()
            transformation_entity.save()
            raise
        else:
            transformation_entity.record.status = RecordStatus.SENT
            transformation_entity.record.save()
            transformation_entity.save()
            return transformation_entity.uuid
