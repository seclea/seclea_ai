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
        self._complete_event = Event()
        self._store_started = Event()
        self._send_started = Event()
        self._send_completed = Event()
        self._store_completed = Event()
        self._store_thread = ProcessorThread(
            processor=Writer,
            name="Store",
            settings=settings,
            input_q=self._store_q,
            started=self._store_started,
            stop=self._stop_event,
            complete=self._complete_event,
            completed=self._store_completed,
            debounce_interval_ms=5000,
        )
        self._send_thread = ProcessorThread(
            processor=Sender,
            name="Send",
            settings=settings,
            input_q=self._send_q,
            started=self._send_started,
            stop=self._stop_event,
            complete=self._complete_event,
            completed=self._send_completed,
            debounce_interval_ms=5000,
        )
        self._store_thread.start()
        self._send_thread.start()
        self._check_started()

    def __del__(self):
        # TODO check status better? vary the kill speeds in some situations?
        self.terminate()

    def terminate(self):
        self._stop_event.set()

    def complete(self):
        # set complete and stop events to trigger the completion in threads
        self._complete_event.set()
        self._stop_event.set()
        # wait until both completed to release this class to gc - which triggers terminate.
        while not (self._send_completed.is_set() and self._store_completed.is_set()):
            time.sleep(0.05)

    def store_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        if entity_dict.get("model_manager", None) == ModelManagers.TENSORFLOW:
            self._save_model_state(**entity_dict)
        else:
            self._store_q.put(entity_dict)

    def send_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        self._send_q.put(entity_dict)

    def _check_started(self, timeout=5.0):
        start = time.time()
        unstarted = {self._store_started, self._store_started}
        to_remove = set()
        while time.time() - start < timeout:
            for idx, ev in enumerate(unstarted):
                if ev.is_set():
                    to_remove.add(ev)
            unstarted = unstarted.difference(to_remove)
            to_remove = set()
            if len(unstarted) == 0:
                return
            print(time.time() - start)
            time.sleep(0.1)
        raise RuntimeError(f"Thread/s not started: {unstarted}")

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
