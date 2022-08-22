import os
import uuid

from pandas import DataFrame

from .processor import Processor
from ...internal.local_db import Record, RecordStatus

# TODO wrap all db requests in transactions to reduce clashes.

"""
Exceptions to handle
- Database errors
"""


class Writer(Processor):
    def __init__(self, settings):
        super().__init__(settings=settings)
        self._settings = settings
        os.makedirs(self._settings["cache_dir"], exist_ok=True)

    def save_record(self, record_id):
        dataset_record = Record.get_by_id(record_id)
        try:
            # TODO take another look at this section.
            path_root = uuid.uuid4()
            dataset_path = self._settings["cache_dir"] / f"{path_root}_tmp.csv"

            # update the record TODO refactor out.
            dataset_record.path = dataset_path
            dataset_record.size = os.path.getsize(dataset_path)
            dataset_record.status = RecordStatus.STORED.value
            dataset_record.save()
        except Exception:
            # update the record TODO refactor out.
            dataset_record.status = RecordStatus.STORE_FAIL.value
            dataset_record.save()
