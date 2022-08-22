import os
import time
import uuid
from abc import ABC
from pathlib import Path
from typing import Dict, List

from pandas import DataFrame
from peewee import SqliteDatabase

from seclea_ai.internal.api.api_interface import PlatformApi as Api
from seclea_ai.internal.local_db import Record, RecordStatus
from seclea_ai.lib.seclea_utils.object_management import Tracked
from typing import Callable
from ..api.base import BaseModelApi


def _assemble_key(record) -> str:
    return f"{record['username']}-{record['project_id']}-{record['entity_id']}"


class Processor(ABC):
    def __init__(self, settings, **kwargs):
        self._settings = settings
        self._db = SqliteDatabase(
            Path.home() / ".seclea" / "seclea_ai.db",
            thread_safe=True,
            pragmas={"journal_mode": "wal"},
        )


class Sender(Processor):
    def __init__(self, settings, api: Api):
        super().__init__(settings=settings)
        self._settings = settings

    def send_record(
            self,
            record_id,
            create_data: dict,
            params: dict,
            api: BaseModelApi
    ):
        """
        :param training_run_name: eg. "Training Run 0"
        :param params: Dict The hyper parameters of the model - can auto extract?
        :return:
        """
        tr_record = Record.get_by_id(record_id)
        try:
            response = api.create(create_data=create_data, params=params)
            tr_record.status = RecordStatus.SENT.value
            tr_record.remote_id = response.uuid
        except ValueError:
            tr_record.status = RecordStatus.SEND_FAIL.value
        tr_record.save()
