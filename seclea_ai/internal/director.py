"""
File storing and uploading data to server
"""
import time
from multiprocessing import Event, Queue
from typing import Dict

from .processors import Sender, Writer
from .threading import ProcessorThread


class Director:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings):
        # setup some defaults
        self._settings = settings
        # TODO probably remove
        ##
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
        self._store_q.put(entity_dict)

    def send_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        self._send_q.put(entity_dict)
