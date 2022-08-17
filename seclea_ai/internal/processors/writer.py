import os
import uuid

from pandas import DataFrame

from .processor import Processor
from ...internal.local_db import Record, RecordStatus
from ...lib.seclea_utils.core import save_object, CompressionFactory
from ...lib.seclea_utils.model_management import ModelManagers, serialize

# TODO wrap all db requests in transactions to reduce clashes.

"""
Exceptions to handle
- Database errors
"""


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
            with open(dataset_path, "rb") as rb:
                comp_path = save_object(
                    rb,
                    file_name=f"{path_root}_compressed",
                    path=self._settings["cache_dir"],
                    compression=CompressionFactory.ZSTD,
                )
            # tidy up intermediate file
            os.remove(dataset_path)

            # update the record TODO refactor out.
            dataset_record.path = comp_path
            dataset_record.size = os.path.getsize(comp_path)
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
        model,
        sequence_num: int,
        model_manager: ModelManagers,
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

            model_data = serialize(model, model_manager)
            save_path = save_object(
                model_data,
                file_name=f"model-{sequence_num}",  # TODO include more identifying info in filename - seclea_ai 798
                path=save_path,
                compression=CompressionFactory.ZSTD,
            )

            record.path = save_path
            record.size = os.path.getsize(save_path)
            record.status = RecordStatus.STORED.value
            record.save()
            return record_id

        except Exception:
            record.status = RecordStatus.STORE_FAIL.value
            record.save()
            raise
