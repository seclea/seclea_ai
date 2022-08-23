import os

from seclea_ai.lib.seclea_utils.object_management import Tracked
from .processor import Processor
from ...internal.local_db import Record, RecordStatus
from seclea_ai.lib.seclea_utils.object_management.mixin import BaseModel

# TODO wrap all db requests in transactions to reduce clashes.

"""
Exceptions to handle
- Database errors
"""


class Writer(Processor):
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)

    def cache_object(self, obj_tr: Tracked, obj_bs: BaseModel):
        """
        saves tracked object into cache dir and updates record.
        @param record_id:
        @return:
        """
        record = Record.get_by_id(obj_bs.uuid)
        try:
            corr_path = os.path.join(self.cache_dir, obj_bs.__class__.__name__, obj_bs.uuid)
            tmp_path = os.path.join(*obj_tr.save_tracked())
            record.path = tmp_path
            record.size = os.path.getsize(obj_tr)
            record.status = RecordStatus.STORED.value
        except Exception as e:
            print(str(e))
            record.status = RecordStatus.STORE_FAIL.value
        record.save()
