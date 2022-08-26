import time
import traceback

from .processor import Processor
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.lib.seclea_utils.object_management.mixin import BaseModel
from ..api.base import BaseModelApi


class Sender(Processor):
    @staticmethod
    def create_object(api: BaseModelApi, obj_bs: BaseModel, params: dict):
        """

        @param api: BaseModelApi used to create object being sent
        @param obj_bs: BaseModel representation of the object (to be serialized)
        @param params: params to be sent with create request
        @param post_kwargs: extra requests post kwargs e.g. files=
        @return:
        """
        tr_record = Record.get_by_id(obj_bs.uuid)
        interval = 0.5
        max_time = 10
        time_passed = 0
        while tr_record.status != RecordStatus.STORED.value and time_passed < max_time:
            print('awaiting store: ', time_passed, ' ', tr_record.status)
            time.sleep(interval)
            time_passed += interval
            tr_record = Record.get_by_id(obj_bs.uuid)

        if time_passed > max_time:
            raise Exception(
                f"Waited to long to store record: record should be {RecordStatus.STORED.value} status is:{tr_record.status}")
        try:
            print(f"Sending object {obj_bs.__class__}: {tr_record.object_ser.keys()}")
            response = api.create(create_data=tr_record.object_ser.copy(), params=params)
            print("response received")
            tr_record.status = RecordStatus.SENT.value
            tr_record.remote_id = response.uuid
        except ValueError as e:
            print(f"send failure, {e}")
            traceback.print_exc()
            tr_record.status = RecordStatus.SEND_FAIL.value
        print("send complete")
        tr_record.save()
