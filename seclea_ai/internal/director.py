"""
File storing and uploading data to server
"""
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
from typing import Dict, List

from .processors import Sender, Writer


class Director:
    """
    Something to wrap backend requests. Maybe use to change the base url??
    """

    def __init__(self, settings, api):
        # setup some defaults
        self._settings = settings
        self._send_q = Queue()
        self.writer = Writer(settings=settings)
        self.sender = Sender(settings=settings, api=api)
        self.threadpool = ThreadPoolExecutor(max_workers=4)
        self.send_executing: List[Future] = list()
        self.write_executing: List[Future] = list()

    def __del__(self):
        self.terminate()

    def terminate(self):
        # cancel any ongoing work.
        for future in self.send_executing:
            future.cancel()
        for future in self.write_executing:
            future.cancel()
        # wait until all resources freed
        self.threadpool.shutdown(wait=True)

    def complete(self):
        # make sure all the entities to be sent have been scheduled
        while not self._send_q.empty():
            time.sleep(0.5)
        # wait for the writes to complete
        for future in self.write_executing:
            future.result()
        # wait for the last send to complete
        for future in self.send_executing:
            future.result()

    def store_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        future = self.threadpool.submit(self.writer.funcs[entity_dict["entity"]], **entity_dict)
        future.add_done_callback(self._write_completed)
        self.write_executing.append(future)

    def send_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        # only put in the queue - if we schedule directly it causes ordering issues.
        if len(self.send_executing) == 0:
            future = self.threadpool.submit(self.sender.funcs[entity_dict["entity"]], **entity_dict)
            future.add_done_callback(self._send_completed)
            self.send_executing.append(future)
        else:
            self._send_q.put(entity_dict)

    def _write_completed(self, future) -> None:
        # TODO deal with return value.
        result = future.result()
        print(f"Write completed - {result}")
        self.write_executing.remove(future)

    def _send_completed(self, future) -> None:
        # here we use a queue to guarantee the ordering and avoid dependency issues.
        # TODO add handling future return/exception
        result = future.result()
        print(f"Send completed - {result}")
        self.send_executing.remove(future)
        try:
            entity_dict = self._send_q.get(False)
        except queue.Empty:
            return
        else:
            future = self.threadpool.submit(self.sender.funcs[entity_dict["entity"]], **entity_dict)
            future.add_done_callback(self._send_completed)
            self.send_executing.append(future)