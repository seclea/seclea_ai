import os

from seclea_ai.lib.seclea_utils.object_management import Tracked
from .processor import Processor
from ...internal.local_db import Record, RecordStatus
from seclea_ai.lib.seclea_utils.object_management.mixin import BaseModel
import traceback
from ..api.base import BaseModelApi

# TODO wrap all db requests in transactions to reduce clashes.

"""
Exceptions to handle
- Database errors
"""


class Writer(Processor):
    def __init__(self, cache_dir: str):
        super().__init__(cache_dir)

    def cache_object(self, obj_tr: Tracked, obj_bs: BaseModel, api: BaseModelApi):
        """
        saves tracked object into cache dir and updates record.
        @param record_id:
        @return:
        """
        print("CACHING OBJECT")
        record = Record.get_by_id(obj_bs.uuid)
        try:
            corr_path = os.path.join(self.cache_dir, obj_bs.__class__.__name__, str(obj_bs.uuid))
            path = obj_tr.save_tracked(corr_path)
            record.path = os.path.join(*path)
            record.size = os.path.getsize(record.path)
            record.status = RecordStatus.STORED.value
            # TODO: this along with polling for saving to be completed by the sender is an architectural design flaw,
            #  leaving this for now but must be addressed as the stem from the same core issue.
            for key in api.file_keys:
                record.object_ser.update({key: record.path})
        except Exception as e:
            print("Cache error ", str(e))
            traceback.print_exc()
            record.status = RecordStatus.STORE_FAIL.value
        print("CACHE COMPLETE")
        record.save()
