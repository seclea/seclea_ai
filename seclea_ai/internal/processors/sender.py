from .processor import Processor
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.lib.seclea_utils.object_management.mixin import BaseModel
from ..api.base import BaseModelApi


class Sender(Processor):
    @staticmethod
    def create_object(api: BaseModelApi, obj_bs: BaseModel, params: dict, **post_kwargs):
        """

        @param api: BaseModelApi used to create object being sent
        @param obj_bs: BaseModel representation of the object (to be serialized)
        @param params: params to be sent with create request
        @param post_kwargs: extra requests post kwargs e.g. files=
        @return:
        """
        tr_record = Record.get_by_id(obj_bs.uuid)
        if tr_record.status != RecordStatus.IN_MEMORY:
            raise Exception(f"record should be in memeory status is:{tr_record.status}")
        try:
            response = api.create(create_data=obj_bs.serialize(), params=params, **post_kwargs)

            tr_record.status = RecordStatus.SENT.value
            tr_record.remote_id = response.uuid
        except ValueError:
            tr_record.status = RecordStatus.SEND_FAIL.value
        tr_record.save()
