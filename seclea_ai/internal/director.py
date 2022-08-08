"""
File storing and uploading data to server
"""
import logging
import queue
import time
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue
from typing import Dict, List, Any

from .processors.sender import Sender
from .processors.writer import Writer

logger = logging.getLogger("seclea_ai")


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
        self.errors = list()

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
        # check for errors and throw if there are any
        self._check_and_throw()
        future = self.threadpool.submit(self.writer.funcs[entity_dict["entity"]], **entity_dict)
        self.write_executing.append(future)
        future.add_done_callback(self._write_completed)

    def send_entity(self, entity_dict: Dict) -> None:  # TODO add return for status
        # check for errors and throw if there are any
        self._check_and_throw()
        # put in queue for sending - directly submit if no send executing to start the chain with callbacks
        if len(self.send_executing) == 0:
            future = self.threadpool.submit(self.sender.funcs[entity_dict["entity"]], **entity_dict)
            self.send_executing.append(future)
            future.add_done_callback(self._send_completed)
        else:
            self._send_q.put(entity_dict)

    def _write_completed(self, future) -> None:
        try:
            result = future.result()
        except Exception as e:
            self.errors.append(e)
        else:
            print(f"Write completed - {result}")
        finally:
            self.write_executing.remove(future)

    def _send_completed(self, future) -> None:
        # here we use a queue to guarantee the ordering and minimise dependency issues.
        try:
            result = future.result()
        except Exception as e:
            self.errors.append(e)
            return  # need to return otherwise second try except block runs.
        else:
            print(f"Send completed - {result}")
        finally:
            self.send_executing.remove(future)

        try:
            entity_dict = self._send_q.get(False)
        except queue.Empty:
            return
        else:
            future = self.threadpool.submit(self.sender.funcs[entity_dict["entity"]], **entity_dict)
            future.add_done_callback(self._send_completed)
            self.send_executing.append(future)

    def _check_and_throw(self):
        """Check the error queue and throw any errors in there. Needed for user thread to deal with them"""
        if len(self.errors) == 0:
            return
        # make sure the errors are printed so we don't miss any
        for error in self.errors:
            logger.error(error)
        # we can only raise one so we raise the first one in the list.
        raise self.errors[0]

    def _deal_with_errors(self, future, queue) -> Any:
        """Extracted method for dealing with runtime errors in the threads"""
