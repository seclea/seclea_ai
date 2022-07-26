"""
File storing and uploading data to server
"""
import os
import time
from multiprocessing import Event, Queue
from pathlib import Path
from typing import Dict

from peewee import SqliteDatabase

from .local_db import RecordStatus, Record
from .processors import Sender, Writer
from .threading import ProcessorThread
from ..lib.seclea_utils.core import CompressionFactory, save_object
from ..lib.seclea_utils.model_management import ModelManagers, serialize


class Director:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings):
        # setup some defaults
        self._settings = settings
        # TODO probably remove
        ##
        self._db = SqliteDatabase(Path.home() / ".seclea" / "seclea_ai.db", thread_safe=True)
        self._store_q = Queue()
        self._send_q = Queue()
        self._stop_event = Event()
        self._store_thread = ProcessorThread(
            processor=Writer,
            name="Store",
            settings=settings,
            input_q=self._store_q,
            stop=self._stop_event,
            debounce_interval_ms=5000,
        )
        self._send_thread = ProcessorThread(
            processor=Sender,
            name="Send",
            settings=settings,
            input_q=self._send_q,
            stop=self._stop_event,
            debounce_interval_ms=5000,
        )
        self._store_thread.start()
        self._send_thread.start()

    def terminate(self):
        # signal threads to finish - then wait and join them
        self._stop_event.set()
        time.sleep(2)  # TODO find a better way to wait.
        self._store_thread.join()
        self._send_thread.join()

    def complete(self):
        # wait until send_q is empty
        while not self._send_q.empty():
            time.sleep(0.1)
        self.terminate()

    def store_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        if entity_dict.get("model_manager", None) == ModelManagers.TENSORFLOW:
            self._save_model_state(**entity_dict)
        else:
            self._store_q.put(entity_dict)

    def send_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        self._send_q.put(entity_dict)

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
        self._db.connect()
        record = Record.get_by_id(record_id)
        try:
            training_run_id = record.dependencies[0]
        except IndexError:
            raise ValueError(
                "Training run must be uploaded before model state something went wrong"
            )
        finally:
            self._db.close()
        try:
            # TODO look again at this.
            save_path = (
                self._settings["cache_dir"]
                / f"{self._settings['project_name']}"
                / f"{str(training_run_id)}"
            )
            os.makedirs(save_path, exist_ok=True)

            model_data = serialize(model, model_manager)
            save_path = save_object(
                model_data,
                file_name=f"model-{sequence_num}",  # TODO include more identifying info in filename - seclea_ai 798
                path=save_path,
                compression=CompressionFactory.ZSTD,
            )

            record.path = save_path
            record.status = RecordStatus.STORED.value
            record.save()

        except Exception as e:
            record.status = RecordStatus.STORE_FAIL.value
            record.save()
            print(e)
        finally:
            self._db.close()
