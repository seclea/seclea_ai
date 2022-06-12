"""
File storing and uploading data to server
"""
import time
from enum import Enum
from multiprocessing import Event, Queue
from typing import Dict

from .local_db import MyDatabase
from .processors import Sender, Writer
from .threading import ProcessorThread


class RecordStatus(Enum):
    IN_MEMORY = "in_memory"
    STORED = "stored"
    SENT = "sent"
    STORE_FAIL = "store_fail"
    SEND_FAIL = "send_fail"


class FileProcessor:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings, transmission, auth_service):
        # setup some defaults
        self._settings = settings
        # TODO probably remove
        self._transmission = transmission
        self._auth_service = auth_service
        ##
        self._dbms = MyDatabase()
        self._store_q = Queue()
        self._send_q = Queue()
        self._stop = Event()
        self._store_thread = ProcessorThread(
            processor=Writer,
            name="Store",
            settings=settings,
            input_q=self._store_q,
            stop=self._stop,
            debounce_interval_ms=5000,
        )
        self._send_thread = ProcessorThread(
            processor=Sender,
            name="Send",
            settings=settings,
            input_q=self._send_q,
            stop=self._stop,
            debounce_interval_ms=5000,
        )
        self._store_thread.start()
        self._send_thread.start()

    def stop(self):
        # signal threads to finish - then wait and join them
        self._stop.set()
        time.sleep(2)  # TODO find a better way to wait.
        self._store_thread.join()
        self._send_thread.join()

    def store_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        self._store_q.put(entity_dict)

    def send_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        self._send_q.put(entity_dict)
