import os
from typing import Dict

from .processor import Processor
from ..persistence.models import Dataset, ModelState
from ..schemas import ModelStateSchema, DatasetSchema
from ...internal.persistence.record import RecordStatus

# TODO wrap all db requests in transactions to reduce clashes.
from ...lib.seclea_utils.object_management import Tracked

"""
Exceptions to handle
- Database errors
"""


class Writer(Processor):
    def __init__(self, settings):
        super().__init__(settings=settings)
        self._settings = settings
        self.func_map = {
            DatasetSchema: self._save_dataset,
            ModelStateSchema: self._save_model_state,
        }

    def _save_dataset(
        self,
        dataset_schema: Dict,
    ):
        """
        Save dataset in local temp directory
        """
        # validate input data
        dataset_schema = DatasetSchema.parse_obj(dataset_schema)

        # TODO add db/connection management
        dataset_entity = Dataset.get(Dataset.uuid == dataset_schema.uuid)

        try:
            os.makedirs(self._settings["cache_dir"], exist_ok=True)

            # TODO - need to ensure that underlying saving logic is thread safe.
            dataset = Tracked(
                dataset_schema.dataset, cleanup=True
            )  # cleanup removes the intermediate directory after object cleaned.
            dataset.object_manager.full_path = self._settings["cache_dir"], str(dataset_schema.uuid)
            dataset_file_path = os.path.join(
                *dataset.save_tracked(path=self._settings["cache_dir"])
            )

            # update the record TODO refactor out.
            dataset_entity.dataset = dataset_file_path
            dataset_entity.record.size = os.path.getsize(dataset_file_path)
            dataset_entity.record.status = RecordStatus.STORED
            dataset_entity.record.save()
            dataset_entity.save()
            return dataset_entity.uuid

        except Exception:
            # update the record TODO refactor out.
            dataset_entity.record.status = RecordStatus.STORE_FAIL
            dataset_entity.save()
            raise

    def _save_model_state(
        self,
        model_state_schema: Dict,
        **kwargs,
    ):
        """
        Save model state in local temp directory
        """
        model_state_schema = ModelStateSchema.parse_obj(model_state_schema)

        model_state_entity = ModelState.get(ModelState.uuid == model_state_schema.uuid)

        try:
            # TODO look again at this.
            save_path = self._settings["cache_dir"] / str(model_state_schema.training_run.uuid)
            os.makedirs(save_path, exist_ok=True)

            # TODO check underlying save logic is thread safe
            model = Tracked(model_state_schema.state)
            file_name = f"model-{model_state_schema.sequence_num}"
            model.object_manager.full_path = save_path, file_name

            save_path = os.path.join(*model.save_tracked(path=save_path))

            model_state_entity.state = save_path
            model_state_entity.record.size = os.path.getsize(save_path)
            model_state_entity.record.status = RecordStatus.STORED
            model_state_entity.record.save()
            model_state_entity.save()
            return model_state_entity.id

        except Exception:
            model_state_entity.record.status = RecordStatus.STORE_FAIL
            model_state_entity.save()
            raise
