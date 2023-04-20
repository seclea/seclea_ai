import os
import uuid

from pandas import DataFrame

from .processor import Processor
from ...internal.persistence.record import Record, RecordStatus, RecordEntity

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
        self.funcs = {
            RecordEntity.DATASET: self._save_dataset,
            RecordEntity.MODEL_STATE: self._save_model_state,
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

            # TODO - need to ensure that underlying saving logic is thread safe.
            dataset = Tracked(
                dataset, cleanup=True
            )  # cleanup removes the intermediate directory after object cleaned.
            dataset.object_manager.full_path = self._settings["cache_dir"], uuid.uuid4().__str__()
            dataset_file_path = os.path.join(
                *dataset.save_tracked(path=self._settings["cache_dir"])
            )

            # update the record TODO refactor out.
            dataset_record.path = dataset_file_path
            dataset_record.size = os.path.getsize(dataset_file_path)
            dataset_record.status = RecordStatus.STORED
            dataset_record.save()
            return record_id

        except Exception:
            # update the record TODO refactor out.
            dataset_record.status = RecordStatus.STORE_FAIL
            dataset_record.save()
            raise

    def _save_model_state(
        self,
        record_id,
        model,
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

            # TODO check underlying save logic is thread safe
            model = Tracked(model)
            file_name = f"model-{sequence_num}"
            model.object_manager.full_path = save_path, file_name

            save_path = os.path.join(*model.save_tracked(path=save_path))

            record.path = save_path
            record.size = os.path.getsize(save_path)
            record.status = RecordStatus.STORED
            record.save()
            return record_id

        except Exception:
            record.status = RecordStatus.STORE_FAIL
            record.save()
            raise
